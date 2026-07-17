import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image
from transformers import BatchEncoding

from train import (
    _validate_camera_checkpoint_schema,
    compatible_clip_checkpoint_key,
    connector_checkpoint_schema,
    diffusion_training_step,
    DiffusionPhaseSpec,
    DiffusionStepOutput,
    estimate_steps,
    filter_clip_checkpoint_incompatibilities,
    make_encode_gemma,
    residual_context_for_captions,
    resolve_autocast_dtype,
    resolve_model_weight_dtype,
    run_diffusion_training_loop,
    run_sara_training,
    select_training_captions,
    validate_connector_checkpoint_schema,
    validate_residual_sara_resume,
)
from pure_ella.config import TrainConfig, resolve_sd_checkpoint
from pure_ella.camera import (
    CAMERA_CONDITION_DIM,
    CameraConditioner,
    apply_camera_dropout,
    camera_condition_schema,
    camera_conditioned_unet,
    camera_metadata_to_tensor,
    extract_camera_metadata,
    install_camera_conditioner,
    make_camera_condition,
)
from pure_ella.connector import build_connector
from pure_ella.dataset import (
    BucketBatchDataset,
    RoundRobinSources,
    StreamingSDDataset,
    _local_parquet_files,
    resize_full_frame,
    resize_long_edge,
)
from pure_ella.prompts import (
    PromptTextDataset,
    make_prompt_dataloader,
    sample_validation_prompts,
)
from pure_ella.diagnostics import (
    ClipGeometryLoss,
    generate_case_image,
    guided_prediction,
    suffix_counterfactual_sensitivity,
    validate_suffix_counterfactual_token_boundaries,
)
from scripts.audit_camera_metadata import classify_geometry_record
from scripts.evaluate_residual_paired import (
    exact_sign_test_two_sided,
    masked_per_sample_mse,
)
from pure_ella.sara import (
    build_sara_attn2_kv_sparse_masks,
    capture_sara_selected_values,
    install_sara_gradient_masks,
    remove_sara_gradient_masks,
    sara_selected_delta_metrics,
)
from pure_ella.longclip_sara import (
    LONGCLIP_L_CONTEXT_TOKENS,
    LONGCLIP_L_WIDTH,
    LongClipSaraConfig,
    install_longclip_sara_sidecars,
    is_longclip_context_overflow,
    load_longclip_sara_checkpoint,
    longclip_attention_mask,
    longclip_sara_schema,
    validate_longclip_sara_checkpoint,
)
from pure_ella.p4 import P4DeltaBlock, apply_axial_2d_rope, p4_state_dict
from pure_ella.resolution import (
    ResolutionConditioner,
    make_resolution_condition,
    make_resolution_condition_from_latents,
)
from scripts.train_longclip_sara import (
    bucket_keys_for_batch,
    fixed_validation_timesteps,
    heldout_diffusion_loss,
    masked_per_sample_mse as longclip_masked_per_sample_mse,
    relative_prediction_rms,
    validation_skip_reason,
)


class ConnectorTrainingTests(unittest.TestCase):
    def test_longclip_mask_uses_eot_position_not_nonzero_tokens(self):
        tokens = torch.tensor([
            [49406, 0, 12, 49407, 0, 0],
            [49406, 42, 49407, 0, 0, 0],
        ])
        mask = longclip_attention_mask(tokens)
        self.assertEqual(mask.tolist(), [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ])
        self.assertTrue(is_longclip_context_overflow(RuntimeError(
            "Input x is too long for context length 248")))
        self.assertFalse(is_longclip_context_overflow(RuntimeError("CUDA failed")))

    def test_longclip_validation_is_fixed_and_preserves_global_rng(self):
        cfg = LongClipSaraConfig(
            sd_checkpoint="stylejourney.safetensors",
            longclip_repo="repo", longclip_checkpoint="model.pt",
            output_dir="output", data_sources=["train.parquet"],
            validation_data_sources=["validation.parquet"],
            validation_max_samples=2, train_batch_size=2,
            aspect_ratio_buckets=[(64, 64)], max_image_dimension=64,
        )

        class Distribution:
            def __init__(self, value): self.value = value
            def mode(self): return self.value

        class VAE:
            config = SimpleNamespace(scaling_factor=1.0)
            def encode(self, image):
                return SimpleNamespace(latent_dist=Distribution(image))

        class Scheduler:
            config = SimpleNamespace(num_train_timesteps=1000)
            def add_noise(self, latent, noise, timestep):
                return latent + noise

        class UNet(nn.Module):
            def forward(self, sample, timestep, encoder_hidden_states,
                        class_labels=None):
                return SimpleNamespace(sample=torch.zeros_like(sample))

        state = SimpleNamespace(
            cfg=cfg, device=torch.device("cpu"), unet_dtype=torch.float32,
            autocast_dtype=None, unet=UNet(), vae=VAE(), scheduler=Scheduler(),
        )
        batch = {
            "caption": ["first long caption", "second long caption"],
            "image": torch.zeros(2, 4, 2, 2),
            "bucket": torch.tensor([640, 1024]),
        }
        encoder = SimpleNamespace(encode=lambda captions, **kwargs: (
            torch.zeros(len(captions), 248, 768),
            torch.ones(len(captions), 248, dtype=torch.long)))
        torch.manual_seed(919)
        before = torch.random.get_rng_state().clone()
        with patch(
                "scripts.train_longclip_sara.make_streaming_dataloader",
                return_value=[batch]):
            first = heldout_diffusion_loss(state, cfg, encoder)
            middle = torch.random.get_rng_state().clone()
            second = heldout_diffusion_loss(state, cfg, encoder)
            after = torch.random.get_rng_state().clone()

        self.assertEqual(first, second)
        self.assertEqual(
            state.longclip_validation_bucket_summary["buckets"]["640x1024"]["count"],
            2)
        self.assertTrue(torch.equal(before, middle))
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(
            fixed_validation_timesteps(5, 1000, torch.device("cpu")).tolist(),
            [0, 250, 500, 749, 999])
        cfg.validation_max_samples = 0
        self.assertEqual(validation_skip_reason(cfg), "validation_max_samples=0")

    def test_longclip_validation_is_independent_of_batch_grouping(self):
        prediction = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[2.0, 4.0], [6.0, 8.0]]],
        ])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
        ])
        together = longclip_masked_per_sample_mse(
            prediction, target, mask)
        separate = torch.cat([
            longclip_masked_per_sample_mse(
                prediction[index:index + 1], target[index:index + 1],
                mask[index:index + 1])
            for index in range(2)
        ])
        self.assertTrue(torch.equal(together, separate))
        self.assertAlmostEqual(float(together.mean()), float(separate.mean()))

    def test_longclip_validation_requires_the_declared_sample_count(self):
        cfg = LongClipSaraConfig(
            sd_checkpoint="stylejourney.safetensors",
            longclip_repo="repo", longclip_checkpoint="model.pt",
            output_dir="output", data_sources=["train.parquet"],
            validation_data_sources=["validation.parquet"],
            validation_max_samples=2, train_batch_size=1,
            aspect_ratio_buckets=[(64, 64)], max_image_dimension=64,
        )
        class Distribution:
            def mode(self): return torch.zeros(1, 4, 2, 2)
        state = SimpleNamespace(
            cfg=cfg, device=torch.device("cpu"), unet_dtype=torch.float32,
            autocast_dtype=None,
            vae=SimpleNamespace(
                config=SimpleNamespace(scaling_factor=1.0),
                encode=lambda image: SimpleNamespace(latent_dist=Distribution())),
            scheduler=SimpleNamespace(
                config=SimpleNamespace(num_train_timesteps=1000),
                add_noise=lambda latent, noise, timestep: latent + noise),
            unet=nn.Identity(),
        )
        # The count check occurs after model evaluation, so reuse the small
        # fake from the deterministic test via a one-sample functional UNet.
        state.unet = SimpleNamespace(
            training=True, eval=lambda: None, train=lambda mode=True: None)
        batch = {
            "caption": ["only one"],
            "image": torch.zeros(1, 4, 2, 2),
            "bucket": (512, 512),
        }
        encoder = SimpleNamespace(encode=lambda captions, **kwargs: (
            torch.zeros(len(captions), 248, 768),
            torch.ones(len(captions), 248, dtype=torch.long)))
        with patch(
                "scripts.train_longclip_sara.make_streaming_dataloader",
                return_value=[batch]), patch(
                "scripts.train_longclip_sara.camera_conditioned_unet",
                return_value=SimpleNamespace(sample=torch.zeros(1, 4, 2, 2))):
            with self.assertRaisesRegex(RuntimeError, "expected 2 samples"):
                heldout_diffusion_loss(state, cfg, encoder)

    def test_longclip_short_prompt_relative_rms(self):
        baseline = torch.ones(2, 3)
        self.assertEqual(relative_prediction_rms(baseline, baseline), 0.0)
        self.assertAlmostEqual(
            relative_prediction_rms(baseline * 1.1, baseline), 0.1, places=6)

    def test_longclip_sara_config_keeps_the_native_direct_contract(self):
        cfg = LongClipSaraConfig(
            sd_checkpoint="stylejourney.safetensors",
            longclip_repo="/tmp/Long-CLIP",
            longclip_checkpoint="/tmp/longclip-L.pt",
            output_dir="output", data_sources=["train.parquet"],
        )
        self.assertEqual(cfg.context_tokens, LONGCLIP_L_CONTEXT_TOKENS)
        self.assertEqual(cfg.longclip_width, LONGCLIP_L_WIDTH)
        self.assertEqual(cfg.gradient_accumulation_steps, 2)
        self.assertEqual(len(cfg.complex_generation_cases), 4)
        with self.assertRaisesRegex(ValueError, "context_tokens=248"):
            LongClipSaraConfig(
                sd_checkpoint="stylejourney.safetensors",
                longclip_repo="repo", longclip_checkpoint="model.pt",
                output_dir="output", data_sources=["train.parquet"],
                context_tokens=77,
            )

    def test_resolution_conditioner_zero_init_and_latent_features(self):
        conditioner = ResolutionConditioner(output_dim=6, hidden_dim=8)
        labels = make_resolution_condition(1024, 512, batch_size=2)
        self.assertEqual(labels.tolist(), [[0.0, -1.0], [0.0, -1.0]])
        latent_labels = make_resolution_condition_from_latents(
            torch.zeros(3, 4, 80, 128))
        self.assertEqual(latent_labels.shape, (3, 2))
        self.assertAlmostEqual(float(latent_labels[0, 0]), 0.0)
        self.assertAlmostEqual(float(latent_labels[0, 1]), -0.6780719)
        output = conditioner(labels)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_p4_rope_contract_and_shape(self):
        q = torch.randn(2, 4, 12, 8)
        k = torch.randn(2, 4, 12, 8)
        qr, kr = apply_axial_2d_rope(q, k, h=3, w=4, base=10000.0)
        self.assertEqual(qr.shape, q.shape)
        self.assertEqual(kr.shape, k.shape)
        self.assertFalse(torch.equal(qr, q))
        with self.assertRaisesRegex(ValueError, "head_dim % 4"):
            apply_axial_2d_rope(
                torch.randn(1, 2, 4, 6), torch.randn(1, 2, 4, 6), h=2, w=2)

    def test_p4_block_exact_zero_gate_identity_and_gradients(self):
        torch.manual_seed(123)
        block = P4DeltaBlock(
            channels=8, time_embed_dim=16, hidden_dim=16, heads=4,
            ff_mult=1.5, use_rope=True, site="pre_mid")
        x = torch.randn(2, 8, 3, 4, requires_grad=True)
        temb = torch.randn(2, 16)
        out = block(x, temb)
        self.assertTrue(torch.equal(out, x))
        self.assertGreater(block.last_branch_rms, 0.0)

        probe = torch.randn_like(out)
        (out * probe).sum().backward()
        gate_grad = block.gate.grad
        self.assertIsNotNone(gate_grad)
        assert gate_grad is not None
        self.assertGreater(abs(float(gate_grad)), 0.0)

        block.zero_grad(set_to_none=True)
        x.grad = None
        with torch.no_grad():
            block.gate.fill_(0.1)
        out_open = block(x, temb)
        (out_open * probe).sum().backward()
        branch_grad_norm = sum(
            float(param.grad.detach().abs().sum())
            for name, param in block.named_parameters()
            if name != "gate" and param.grad is not None
        )
        self.assertGreater(branch_grad_norm, 0.0)

    def test_longclip_config_validates_p4_rope_head_dim(self):
        LongClipSaraConfig(
            sd_checkpoint="stylejourney.safetensors",
            longclip_repo="/tmp/Long-CLIP",
            longclip_checkpoint="/tmp/longclip-L.pt",
            output_dir="output", data_sources=["train.parquet"],
            p4_enabled=True, p4_variant="rope", p4_hidden_dim=1280,
            p4_heads=8,
        )
        with self.assertRaisesRegex(ValueError, "per-head dim divisible by 4"):
            LongClipSaraConfig(
                sd_checkpoint="stylejourney.safetensors",
                longclip_repo="/tmp/Long-CLIP",
                longclip_checkpoint="/tmp/longclip-L.pt",
                output_dir="output", data_sources=["train.parquet"],
                p4_enabled=True, p4_variant="rope", p4_hidden_dim=18,
                p4_heads=3,
            )

    def test_longclip_config_rejects_resolution_dropout(self):
        with self.assertRaisesRegex(ValueError, "resolution_conditioning_dropout_prob must be 0.0"):
            LongClipSaraConfig(
                sd_checkpoint="stylejourney.safetensors",
                longclip_repo="/tmp/Long-CLIP",
                longclip_checkpoint="/tmp/longclip-L.pt",
                output_dir="output", data_sources=["train.parquet"],
                resolution_conditioning_enabled=True,
                resolution_conditioning_dropout_prob=0.1,
            )

    def test_longclip_sidecar_checkpoint_loader_restores_strict_state(self):
        class Down(nn.Module):
            def forward(self, hidden_states, temb=None, **kwargs):
                return hidden_states + 0.25, (hidden_states,)

        class Mid(nn.Module):
            def forward(self, hidden_states, temb=None, **kwargs):
                return hidden_states * 1.5

        class TimeEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_2 = nn.Linear(16, 16)

        class FakeUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(block_out_channels=[8])
                self.time_embedding = TimeEmbedding()
                self.down_blocks = nn.ModuleList([Down()])
                self.mid_block = Mid()
                self.class_embedding = None
                self.anchor = nn.Parameter(torch.zeros(2))

            def forward(self, sample, timestep, encoder_hidden_states,
                        class_labels=None):
                temb = torch.ones(sample.shape[0], 16, dtype=sample.dtype)
                if self.class_embedding is not None:
                    temb = temb + self.class_embedding(class_labels).to(dtype=temb.dtype)
                hidden, _ = self.down_blocks[-1](sample + self.anchor[0], temb=temb)
                hidden = self.mid_block(hidden, temb=temb)
                return SimpleNamespace(sample=hidden)

        class FakeEncoder:
            def provenance(self):
                return {"checkpoint_sha256": "abc", "context_tokens": 248}

        with tempfile.TemporaryDirectory() as directory:
            sd_checkpoint = Path(directory) / "stylejourney.safetensors"
            sd_checkpoint.write_bytes(b"stylejourney")
            cfg = LongClipSaraConfig(
                sd_checkpoint=str(sd_checkpoint), longclip_repo="repo",
                longclip_checkpoint="model.pt", output_dir="output",
                data_sources=["train.parquet"],
                require_short_prompt_regression_gate=False,
                resolution_conditioning_enabled=True,
                p4_enabled=True, p4_variant="no_pe", p4_hidden_dim=16,
                p4_heads=4,
            )
            encoder = FakeEncoder()
            source = FakeUNet()
            install_longclip_sara_sidecars(source, cfg)
            with torch.no_grad():
                source.anchor[0] = 1.25
                source.class_embedding.output.bias.fill_(0.125)
                source.p4_blocks["pre_mid"].gate.fill_(0.2)
            checkpoint = {
                "longclip_sara_schema": longclip_sara_schema(cfg, encoder),
                "sparse_values": {
                    "anchor": {
                        "mask": torch.tensor([True, False]),
                        "values": torch.tensor([1.25]),
                        "shape": (2,),
                    }
                },
                "resolution_conditioner_state_dict": {
                    key: value.detach().clone()
                    for key, value in source.class_embedding.state_dict().items()
                },
                "p4_state_dict": p4_state_dict(source),
                "validation_gate": {
                    "baseline_loss": 1.0,
                    "final_loss": 0.99,
                    "relative_change": -0.01,
                    "heldout_ran": True,
                    "heldout_passed": True,
                    "short_prompt_relative_rms": None,
                    "short_prompt_ran": False,
                    "short_prompt_passed": False,
                    "passed": True,
                },
                "completion": {"completed": True, "optimizer_steps": 1},
                "run_config": cfg.to_dict(),
            }
            target = FakeUNet()
            loaded = load_longclip_sara_checkpoint(
                target, checkpoint, cfg, encoder, "unit")
            self.assertEqual(loaded["sparse_values"], 1)
            self.assertGreater(loaded["sidecars"]["resolution_tensors"], 0)
            self.assertGreater(loaded["sidecars"]["p4_tensors"], 0)
            sample = torch.randn(2, 8, 3, 4)
            timestep = torch.ones(2, dtype=torch.long)
            context = torch.zeros(2, 1, 1)
            labels = make_resolution_condition(1024, 512, batch_size=2)
            source_out = source(
                sample, timestep, encoder_hidden_states=context,
                class_labels=labels).sample
            target_out = target(
                sample, timestep, encoder_hidden_states=context,
                class_labels=labels).sample
            self.assertTrue(torch.allclose(source_out, target_out, atol=0, rtol=0))

            missing_p4 = dict(checkpoint)
            missing_p4.pop("p4_state_dict")
            with self.assertRaisesRegex(RuntimeError, "missing p4_state_dict"):
                load_longclip_sara_checkpoint(
                    FakeUNet(), missing_p4, cfg, encoder, "missing")

            gate_disabled_cfg = LongClipSaraConfig(
                sd_checkpoint=str(sd_checkpoint), longclip_repo="repo",
                longclip_checkpoint="model.pt", output_dir="output",
                data_sources=["train.parquet"],
                require_validation_gate=False,
                require_short_prompt_regression_gate=False,
            )
            gate_disabled_checkpoint = {
                "longclip_sara_schema": longclip_sara_schema(gate_disabled_cfg, encoder),
                "sparse_values": checkpoint["sparse_values"],
                "validation_gate": {"passed": False},
                "completion": {"completed": True, "optimizer_steps": 1},
                "run_config": gate_disabled_cfg.to_dict(),
            }
            load_longclip_sara_checkpoint(
                FakeUNet(), gate_disabled_checkpoint, gate_disabled_cfg,
                encoder, "gate-disabled")
            with self.assertRaisesRegex(RuntimeError, "did not pass validation gates"):
                load_longclip_sara_checkpoint(
                    FakeUNet(), gate_disabled_checkpoint, gate_disabled_cfg,
                    encoder, "gate-forced", require_validation_gate=True)

            synthetic_shell = dict(gate_disabled_checkpoint)
            synthetic_shell["validation_gate"] = {"passed": True}
            with self.assertRaisesRegex(RuntimeError, "held-out validation"):
                load_longclip_sara_checkpoint(
                    FakeUNet(), synthetic_shell, gate_disabled_cfg,
                    encoder, "synthetic-heldout",
                    require_validation_gate=True,
                    require_short_prompt_regression_gate=False)
            with self.assertRaisesRegex(RuntimeError, "short-prompt validation"):
                load_longclip_sara_checkpoint(
                    FakeUNet(), synthetic_shell, gate_disabled_cfg,
                    encoder, "synthetic-short",
                    require_validation_gate=False,
                    require_short_prompt_regression_gate=True)

            legacy_real_gate = dict(gate_disabled_checkpoint)
            legacy_real_gate["validation_gate"] = {
                "baseline_loss": 1.0,
                "final_loss": 0.99,
                "relative_change": -0.01,
                "short_prompt_relative_rms": 0.01,
                "passed": True,
            }
            load_longclip_sara_checkpoint(
                FakeUNet(), legacy_real_gate, gate_disabled_cfg,
                encoder, "legacy-real",
                require_validation_gate=True,
                require_short_prompt_regression_gate=True)

    def test_bucket_keys_for_batch_requires_single_bucket_pair(self):
        self.assertEqual(
            bucket_keys_for_batch({"bucket": (640, 1024)}, 2),
            ["640x1024", "640x1024"])
        self.assertEqual(
            bucket_keys_for_batch({"bucket": torch.tensor([512, 768])}, 1),
            ["512x768"])
        with self.assertRaisesRegex(ValueError, "missing emitted bucket"):
            bucket_keys_for_batch({}, 1)
        with self.assertRaisesRegex(ValueError, r"shape \[2\]"):
            bucket_keys_for_batch({"bucket": torch.tensor([[512, 768]])}, 1)
        with self.assertRaisesRegex(ValueError, r"single \(width, height\) pair"):
            bucket_keys_for_batch({"bucket": [(512, 768), (640, 1024)]}, 2)

    def test_longclip_sara_schema_pins_encoder_and_sparse_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            sd_checkpoint = Path(directory) / "stylejourney.safetensors"
            sd_checkpoint.write_bytes(b"stylejourney")
            cfg = LongClipSaraConfig(
                sd_checkpoint=str(sd_checkpoint), longclip_repo="repo",
                longclip_checkpoint="model.pt", output_dir="output",
                data_sources=["train.parquet"], sara_target_fraction=0.05,
            )
            class FakeEncoder:
                def provenance(self):
                    return {"checkpoint_sha256": "abc", "context_tokens": 248}
            schema = longclip_sara_schema(cfg, FakeEncoder())
            self.assertEqual(schema["conditioning_backend"], "longclip_l_direct")
            self.assertEqual(schema["longclip"]["checkpoint_sha256"], "abc")
            self.assertEqual(schema["sara_target_fraction"], 0.05)
            self.assertEqual(schema["sara_threshold"], 1e-3)
            self.assertIn("sd_checkpoint_sha256", schema)
            checkpoint = {
                "longclip_sara_schema": schema,
                "completion": {"completed": True, "optimizer_steps": 1},
                "sparse_values": {"unet.attn2.to_k.weight": {"values": torch.ones(1)}},
                "validation_gate": {
                    "short_prompt_relative_rms": 0.01,
                    "short_prompt_ran": True,
                    "short_prompt_passed": True,
                    "passed": True,
                },
            }
            validate_longclip_sara_checkpoint(checkpoint, cfg, FakeEncoder(), "test")
            checkpoint["completion"] = {"completed": True, "optimizer_steps": 0}
            with self.assertRaisesRegex(RuntimeError, "zero updates"):
                validate_longclip_sara_checkpoint(checkpoint, cfg, FakeEncoder(), "test")

    def test_longclip_scripts_support_direct_help_invocation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for script in (
                    "scripts/train_longclip_sara.py",
                    "scripts/compare_stylejourney_longclip.py"):
                result = subprocess.run(
                    [sys.executable, str(root / script), "--help"],
                    cwd=directory, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_post_training_complex_cases_keep_canonical_settings(self):
        cases = TrainConfig().complex_generation_cases
        self.assertEqual([case["name"] for case in cases], [
            "warrior_princess",
            "medieval_darth_vader_dragon",
            "goddess_of_death",
            "milkyway",
        ])
        self.assertEqual(cases[0]["sampler"], "dpmpp_sde_karras")
        self.assertEqual(cases[1]["sampler"], "euler_a")
        self.assertEqual(cases[0]["negative_prompt"].startswith("(bonnet)"), True)
        self.assertTrue(all(case["width"] == 768 for case in cases))
        self.assertTrue(all(case["height"] == 960 for case in cases))

    def test_complex_case_generation_applies_and_restores_all_case_settings(self):
        from diffusers import DDPMScheduler, EulerAncestralDiscreteScheduler

        old_scheduler = object()
        state = SimpleNamespace(
            cfg=SimpleNamespace(
                val_steps=11, val_guidance=2.5, val_seed=22,
                complex_prompt_width=512, complex_prompt_height=512),
            scheduler=DDPMScheduler(num_train_timesteps=20),
            inf_scheduler=old_scheduler,
        )
        case = {
            "prompt": "test prompt", "negative_prompt": "no test",
            "sampler": "euler_a", "steps": 20, "guidance": 7.0,
            "seed": 99, "width": 768, "height": 960,
        }
        installed_schedulers = []
        def fake_generate(*args, **kwargs):
            installed_schedulers.append(state.inf_scheduler)
            return "image"

        with patch("pure_ella.diagnostics.generate_ella", side_effect=fake_generate) as generate:
            result = generate_case_image(case, state, context_tokens=77)

        self.assertEqual(result, "image")
        self.assertIs(state.inf_scheduler, old_scheduler)
        self.assertEqual(generate.call_args.args, ("test prompt", state))
        self.assertEqual(generate.call_args.kwargs, {
            "steps": 20, "guidance": 7.0, "seed": 99,
            "negative_prompt": "no test", "height": 960, "width": 768,
            "context_tokens": 77,
        })
        # The requested scheduler was installed while generate_ella ran.
        self.assertIsInstance(installed_schedulers[0], EulerAncestralDiscreteScheduler)

    def test_clip_geometry_can_supervise_all_tokens_and_pool_real_tokens(self):
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        prediction = target.clone()
        prediction[:, 1:] = torch.tensor([[[4.0, -2.0], [-3.0, 5.0]]])
        actual_token_mask = torch.tensor([[1, 0, 0]])
        all_token_mask = torch.ones_like(actual_token_mask)
        loss_fn = ClipGeometryLoss(w_ctr=0.0)

        prefix_only = loss_fn(prediction, target, actual_token_mask)
        full_contract = loss_fn(
            prediction, target, all_token_mask,
            pool_mask=actual_token_mask,
        )

        self.assertAlmostEqual(prefix_only["mse"].item(), 0.0, places=6)
        self.assertGreater(full_contract["mse"].item(), 0.1)
        self.assertAlmostEqual(
            full_contract["pooled_cos"].item(), 1.0, places=6)

    def test_phase0_timestep_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "timestep_sampling"):
            TrainConfig(pretrain_timestep_sampling="random")
        with self.assertRaisesRegex(ValueError, "validation_timesteps"):
            TrainConfig(
                pretrain_num_train_timesteps=1000,
                pretrain_validation_timesteps=[1000],
            )

    def test_fast_prompt_pretrain_requires_prompt_sources(self):
        with self.assertRaisesRegex(ValueError, "pretrain_prompt_sources"):
            TrainConfig(pretrain_text_only_fast=True)

    def test_prompt_only_loader_is_deterministic_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("one\n\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
            kwargs = dict(
                sources=[str(path)], max_samples=4, shuffle=True,
                shuffle_buffer=3, seed=91,
            )
            first = list(PromptTextDataset(**kwargs))
            second = list(PromptTextDataset(**kwargs))

            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertNotIn("", first)

            batches = list(make_prompt_dataloader(
                [str(path)], epoch=0, max_samples=5, batch_size=2,
                shuffle=False, shuffle_buffer=2, base_seed=10,
            ))
            self.assertEqual(
                batches, [["one", "two"], ["three", "four"], ["five"]])

    def test_validation_prompt_sampling_is_fixed_per_source(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.txt"
            second_path = Path(directory) / "second.txt"
            first_path.write_text(
                "".join(f"first {index}\n" for index in range(20)),
                encoding="utf-8",
            )
            second_path.write_text(
                "".join(f"second {index}\n" for index in range(20)),
                encoding="utf-8",
            )
            sources = [str(first_path), str(second_path)]

            sampled = sample_validation_prompts(sources, 5, seed=123)

            self.assertEqual(sampled, sample_validation_prompts(
                sources, 5, seed=123))
            self.assertEqual(set(sampled), {"first", "second"})
            self.assertTrue(all(len(prompts) == 5 for prompts in sampled.values()))

    def test_camera_metadata_extracts_nested_unsplash_exif(self):
        sample = {
            "upstream_json": json.dumps({
                "exif": {
                    "focal_length": "45.0",
                    "aperture_value": "5.0",
                    "iso": "200.0",
                }
            }),
            "caption_detailed": "a mountain photograph",
        }

        metadata = extract_camera_metadata(sample)
        condition = camera_metadata_to_tensor(metadata)

        self.assertEqual(metadata["capture_type"], "photo")
        self.assertIsNone(metadata["vertical_fov_deg"])
        self.assertEqual(metadata["focal_length_mm"], 45.0)
        self.assertEqual(metadata["aperture_f_number"], 5.0)
        self.assertEqual(metadata["iso"], 200.0)
        self.assertEqual(tuple(condition.shape), (CAMERA_CONDITION_DIM,))
        self.assertEqual(condition[4:8].tolist(), [0.0, 1.0, 1.0, 1.0])
        self.assertEqual(condition[-1].item(), 1.0)

    def test_35mm_equivalent_requires_offline_axis_fov(self):
        metadata = extract_camera_metadata({
            "metadata": {"focal_length_35mm": 50},
        })
        self.assertIsNone(metadata["vertical_fov_deg"])

    def test_camera_metadata_merges_all_nested_containers(self):
        metadata = extract_camera_metadata({
            "upstream_json": {"exif": {"focal_length": 35}, "iso": 100},
            "json": json.dumps({"exif": {"aperture": 2.8}}),
            "metadata": {"exif": {"iso": 400}},
        })

        self.assertEqual(metadata["focal_length_mm"], 35.0)
        self.assertEqual(metadata["aperture_f_number"], 2.8)
        self.assertEqual(metadata["iso"], 400.0)

    def test_camera_conditioner_is_exact_identity_at_initialization(self):
        class TinyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.class_embedding = None

            def forward(self, sample, timestep, encoder_hidden_states,
                        class_labels=None):
                output = sample.clone()
                if self.class_embedding is not None:
                    output = output + self.class_embedding(
                        class_labels).reshape(-1, 1, 1, 1)
                return SimpleNamespace(sample=output)

        unet = TinyUNet()
        sample = torch.randn(2, 1, 2, 2)
        timestep = torch.tensor([1, 2])
        context = torch.randn(2, 3, 4)
        baseline = unet(sample, timestep, context).sample
        conditioner = CameraConditioner(
            output_dim=1, hidden_dim=8, fourier_bands=2,
            capture_embed_dim=4)
        install_camera_conditioner(unet, conditioner)
        condition = torch.stack([
            make_camera_condition(vertical_fov_deg=35, capture_type="photo"),
            make_camera_condition(vertical_fov_deg=90, capture_type="photo"),
        ])

        conditioned = camera_conditioned_unet(
            unet, sample, timestep, encoder_hidden_states=context,
            camera_condition=condition).sample

        self.assertTrue(torch.equal(conditioned, baseline))
        self.assertEqual(torch.count_nonzero(conditioner(condition)).item(), 0)
        conditioner(condition).sum().backward()
        self.assertGreater(
            torch.count_nonzero(
                conditioner.geometry_head.mlp[-1].weight.grad).item(), 0)

    def test_camera_conditioner_uses_diffusers_class_embedding_contract(self):
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(16,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=4,
            cross_attention_dim=8,
        ).eval()
        sample = torch.randn(1, 4, 8, 8)
        timestep = torch.tensor([5])
        context = torch.randn(1, 2, 8)
        with torch.no_grad():
            baseline = unet(
                sample, timestep, encoder_hidden_states=context).sample
        conditioner = CameraConditioner(
            unet.time_embedding.linear_2.out_features,
            hidden_dim=8,
            fourier_bands=2,
            capture_embed_dim=4,
        )
        install_camera_conditioner(unet, conditioner)

        with torch.no_grad():
            conditioned = camera_conditioned_unet(
                unet, sample, timestep, encoder_hidden_states=context,
                camera_condition=make_camera_condition(
                    focal_length_mm=35).unsqueeze(0),
            ).sample

        self.assertTrue(torch.equal(conditioned, baseline))

    def test_unknown_camera_condition_remains_zero_after_optimizer_step(self):
        conditioner = CameraConditioner(
            output_dim=3, hidden_dim=8, fourier_bands=2)
        optimizer = torch.optim.AdamW(conditioner.parameters(), lr=0.1)
        known = make_camera_condition(
            vertical_fov_deg=35, capture_type="photo").unsqueeze(0)
        target = torch.ones(1, 3)

        loss = torch.nn.functional.mse_loss(conditioner(known), target)
        loss.backward()
        optimizer.step()

        unknown = conditioner.unknown(2, torch.device("cpu"))
        self.assertTrue(torch.equal(
            conditioner(unknown), torch.zeros(2, 3)))
        self.assertGreater(torch.count_nonzero(conditioner(known)).item(), 0)

    def test_capture_type_is_ignored_until_explicitly_enabled(self):
        conditioner = CameraConditioner(
            output_dim=2, hidden_dim=8, fourier_bands=2,
            use_capture_type=False)
        with torch.no_grad():
            conditioner.geometry_head.mlp[-1].weight.fill_(0.1)
        photo = make_camera_condition(
            vertical_fov_deg=35, capture_type="photo").unsqueeze(0)
        artwork = make_camera_condition(
            vertical_fov_deg=35, capture_type="artwork").unsqueeze(0)

        self.assertTrue(torch.equal(conditioner(photo), conditioner(artwork)))

    def test_camera_scalar_groups_have_separate_parameters(self):
        conditioner = CameraConditioner(
            output_dim=2, hidden_dim=8, fourier_bands=2,
            enable_raw_focal_head=True,
            enable_exposure_head=True,
        )
        condition = make_camera_condition(vertical_fov_deg=35).unsqueeze(0)
        conditioner(condition).sum().backward()

        self.assertGreater(torch.count_nonzero(
            conditioner.geometry_head.mlp[-1].weight.grad).item(), 0)
        for head in (conditioner.raw_focal_head, conditioner.exposure_head):
            gradient = head.mlp[-1].weight.grad
            self.assertTrue(gradient is None or torch.count_nonzero(gradient) == 0)

    def test_camera_checkpoint_schema_rejects_semantic_mismatch(self):
        cfg = TrainConfig()
        cfg.camera_use_capture_type = False
        state = SimpleNamespace(
            cfg=cfg,
            camera_conditioner=CameraConditioner(
                output_dim=2, hidden_dim=8, fourier_bands=2),
        )
        checkpoint = {
            "camera_condition_schema": camera_condition_schema(
                use_capture_type=True),
        }

        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            _validate_camera_checkpoint_schema(checkpoint, state, "test.pt")

    def test_config_rejects_camera_with_active_clip_teacher(self):
        cfg = TrainConfig(
            camera_conditioning_enabled=True,
            camera_experiment_mode="raw_focal_baseline",
            camera_allow_experimental_training=True,
            camera_enable_geometry_head=False,
            camera_enable_raw_focal_head=True,
            use_clip_teacher_delta=True,
            lambda_teacher=1.0,
        )
        with self.assertRaisesRegex(ValueError, "teacher target move"):
            cfg.validate_requested_phases({"ella"})

        diagnostic = TrainConfig(
            run_mode="diagnostic",
            camera_conditioning_enabled=True,
            use_clip_teacher_delta=True,
            lambda_teacher=1.0,
        )
        diagnostic.validate_requested_phases({"ella"})

    def test_config_rejects_dropout_for_camera_only_training(self):
        cfg = TrainConfig(
            camera_conditioning_enabled=True,
            camera_train_connector=False,
            camera_experiment_mode="raw_focal_baseline",
            camera_allow_experimental_training=True,
            camera_enable_geometry_head=False,
            camera_enable_raw_focal_head=True,
            camera_metadata_dropout_prob_ella=0.4,
        )
        with self.assertRaisesRegex(ValueError, "camera-only ELLA"):
            cfg.validate_requested_phases({"ella", "sara"})
        cfg.validate_requested_phases({"sara"})

    def test_camera_training_requires_explicit_mode(self):
        cfg = TrainConfig(camera_conditioning_enabled=True)
        with self.assertRaisesRegex(ValueError, "Camera training is disabled"):
            cfg.validate_requested_phases({"ella"})

        raw = TrainConfig(
            camera_conditioning_enabled=True,
            camera_experiment_mode="raw_focal_baseline",
            camera_enable_geometry_head=False,
            camera_enable_raw_focal_head=True,
        )
        with self.assertRaisesRegex(ValueError, "experimental_training"):
            raw.validate_requested_phases({"ella"})

    def test_guided_prediction_uses_unconditional_plus_scaled_delta(self):
        conditional = torch.tensor([[[[5.0]]]])
        unconditional = torch.tensor([[[[2.0]]]])
        actual = guided_prediction(conditional, unconditional, guidance=4.0)
        self.assertTrue(torch.equal(actual, torch.tensor([[[[14.0]]]])))

    def test_strict_geometry_audit_requires_provenance_conjunction(self):
        sensor_table = {
            "test|camera": {"sensor_width_mm": 36.0, "sensor_height_mm": 24.0}
        }
        resolution = 6000 / 36.0 * 25.4
        exif = {
            "Make": "Test",
            "Model": "Camera",
            "ExifImageWidth": 6000,
            "ExifImageHeight": 4000,
            "FocalPlaneXResolution": resolution,
            "FocalPlaneYResolution": resolution,
            "FocalPlaneResolutionUnit": 2,
            "FocalLength": 50,
            "Orientation": 1,
        }
        eligible = classify_geometry_record(
            (6000, 4000), exif, "false", sensor_table)
        self.assertTrue(eligible["strict_geometry_eligible"])
        self.assertEqual(eligible["dimension_status"], "exact")

        unexplained_swap = classify_geometry_record(
            (4000, 6000), exif, "false", sensor_table)
        self.assertFalse(unexplained_swap["strict_geometry_eligible"])
        self.assertEqual(
            unexplained_swap["dimension_status"], "unexplained_swap")

        exif["Orientation"] = 6
        corrected = classify_geometry_record(
            (4000, 6000), exif, "false", sensor_table)
        self.assertTrue(corrected["strict_geometry_eligible"])
        self.assertEqual(
            corrected["dimension_status"], "orientation_corrected")

        cropped = classify_geometry_record(
            (6000, 4000), exif, "true", sensor_table)
        self.assertFalse(cropped["strict_geometry_eligible"])
        self.assertIn("crop_true", cropped["rejection_reasons"])

        exif["FocalPlaneResolutionUnit"] = 1
        invalid_unit = classify_geometry_record(
            (6000, 4000), exif, "false", sensor_table)
        self.assertFalse(invalid_unit["strict_geometry_eligible"])
        self.assertEqual(invalid_unit["sensor_status"], "not_derivable")

    def test_camera_dropout_replaces_entire_record_with_unknown(self):
        condition = torch.stack([
            make_camera_condition(vertical_fov_deg=35, capture_type="photo"),
            make_camera_condition(iso=800, capture_type="photo"),
        ])
        dropped = apply_camera_dropout(condition, 1.0)
        self.assertEqual(torch.count_nonzero(dropped).item(), 0)

    @staticmethod
    def _tiny_sara_unet():
        class TinyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_a = nn.Module()
                self.block_a.attn2 = nn.Module()
                self.block_a.attn2.to_k = nn.Linear(4, 1, bias=False)
                self.block_b = nn.Module()
                self.block_b.attn2 = nn.Module()
                self.block_b.attn2.to_v = nn.Linear(4, 1, bias=False)
                self.unrelated = nn.Linear(2, 2, bias=False)

        unet = TinyUNet()
        with torch.no_grad():
            unet.block_a.attn2.to_k.weight.copy_(
                torch.tensor([[1.0, 1.0, 1.0, 4.0]])
            )
            unet.block_b.attn2.to_v.weight.copy_(
                torch.tensor([[5.0, 6.0, 7.0, 8.0]])
            )
        return unet

    def test_precision_config_keeps_sara_weights_full_precision(self):
        cfg = TrainConfig(
            model_weight_dtype="float32",
            mixed_precision="bf16",
        )
        self.assertEqual(resolve_model_weight_dtype(cfg), torch.float32)

        with patch("train.torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(
                resolve_autocast_dtype(cfg, torch.device("cuda")),
                torch.bfloat16,
            )

        with self.assertRaisesRegex(ValueError, "mixed_precision"):
            TrainConfig(mixed_precision="fp16")
        with self.assertRaisesRegex(ValueError, "Training requires"):
            TrainConfig(
                model_weight_dtype="bfloat16",
            )

        diagnostic = TrainConfig(
            experiment_stage="stage3_long_context_with_sara",
            run_mode="diagnostic",
            model_weight_dtype="bfloat16",
        )
        self.assertFalse(diagnostic.run_training)
        self.assertTrue(diagnostic.run_sara_phase)

    def test_sara_does_not_train_in_diagnostic_mode(self):
        state = SimpleNamespace(cfg=SimpleNamespace(
            run_training=False,
            run_sara_phase=True,
            sara_epochs=1,
        ))
        with patch(
            "train.build_sara_attn2_kv_sparse_masks"
        ) as build_masks:
            run_sara_training(state)

        build_masks.assert_not_called()

    def test_sara_target_fraction_selects_exact_global_rank_with_ties(self):
        unet = self._tiny_sara_unet()
        summary = build_sara_attn2_kv_sparse_masks(
            unet,
            selection_mode="target_fraction",
            target_fraction=0.25,
            min_sparse_fraction=0.20,
            max_sparse_fraction_warn=0.30,
            max_sparse_fraction_abort=0.50,
        )

        self.assertEqual(summary["selected"], 2)
        self.assertEqual(summary["total_target"], 8)
        self.assertEqual(summary["fraction"], 0.25)
        self.assertEqual(summary["total_unet"], 12)
        self.assertAlmostEqual(summary["target_scope_fraction"], 8 / 12)
        self.assertAlmostEqual(summary["whole_unet_fraction"], 2 / 12)
        self.assertEqual(summary["magnitude_cutoff"], 1.0)
        self.assertEqual(
            unet.block_a.attn2.to_k.weight._sara_sparse_mask.tolist(),
            [[True, True, False, False]],
        )
        self.assertFalse(unet.unrelated.weight.requires_grad)

    def test_sara_threshold_mode_and_minimum_gate_remain_available(self):
        unet = self._tiny_sara_unet()
        summary = build_sara_attn2_kv_sparse_masks(
            unet,
            selection_mode="magnitude_threshold",
            threshold=4.5,
            min_sparse_fraction=0.40,
            max_sparse_fraction_warn=0.60,
            max_sparse_fraction_abort=0.80,
        )
        self.assertEqual(summary["selected"], 4)

        with self.assertRaisesRegex(RuntimeError, "below minimum gate"):
            build_sara_attn2_kv_sparse_masks(
                self._tiny_sara_unet(),
                threshold=1.5,
                min_sparse_fraction=0.50,
                max_sparse_fraction_warn=0.60,
                max_sparse_fraction_abort=0.80,
            )

    def test_sara_reports_selected_weight_delta(self):
        unet = self._tiny_sara_unet()
        build_sara_attn2_kv_sparse_masks(
            unet,
            selection_mode="target_fraction",
            target_fraction=0.25,
            max_sparse_fraction_warn=0.30,
            max_sparse_fraction_abort=0.50,
        )
        baseline = capture_sara_selected_values(unet)
        with torch.no_grad():
            mask = unet.block_a.attn2.to_k.weight._sara_sparse_mask
            unet.block_a.attn2.to_k.weight[mask] += 0.5

        metrics = sara_selected_delta_metrics(unet, baseline)
        self.assertAlmostEqual(metrics["selected_delta_rms"], 0.5)
        self.assertAlmostEqual(metrics["selected_delta_max_abs"], 0.5)
        self.assertAlmostEqual(metrics["selected_delta_relative_l2"], 0.5)

    def test_config_validates_sara_target_fraction_and_gates(self):
        cfg = TrainConfig(
            sara_selection_mode="target_fraction",
            sara_target_fraction=0.10,
            sara_min_sparse_fraction=0.05,
            sara_max_sparse_fraction_warn=0.15,
            sara_max_sparse_fraction_abort=0.25,
        )
        self.assertEqual(cfg.sara_target_fraction, 0.10)
        with self.assertRaisesRegex(ValueError, "between the minimum"):
            TrainConfig(
                sara_selection_mode="target_fraction",
                sara_target_fraction=0.30,
                sara_min_sparse_fraction=0.05,
                sara_max_sparse_fraction_warn=0.15,
                sara_max_sparse_fraction_abort=0.25,
            )

    def test_config_rejects_non_unet_aligned_buckets(self):
        with self.assertRaisesRegex(ValueError, "multiples of 64"):
            TrainConfig(aspect_ratio_buckets=[[520, 392]])

    def test_step_plan_accounts_for_per_bucket_tail_batches(self):
        plan = estimate_steps(
            max_samples=5,
            batch_size=2,
            epochs=1,
            cap=None,
            drop_last=False,
            bucket_count=3,
        )
        self.assertEqual(plan.steps_per_epoch, 4)
        self.assertEqual(plan.effective, 4)

        drop_last_plan = estimate_steps(
            max_samples=5,
            batch_size=2,
            epochs=1,
            cap=None,
            drop_last=True,
            bucket_count=3,
        )
        self.assertEqual(drop_last_plan.steps_per_epoch, 2)

    def test_diffusion_loop_accumulates_and_averages_microbatches(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        cfg = TrainConfig(
            gradient_accumulation_steps=2, validation_every_opt_steps=0,
            generation_grid_every_opt_steps=0, max_samples_ella=4,
            ella_epochs=1, ella_max_opt_steps=2, train_batch_size=1,
            grad_clip_norm=100.0,
        )
        state = SimpleNamespace(cfg=cfg, device=torch.device("cpu"), wandb=None)
        batches = [{"value": value} for value in (1.0, 3.0, 5.0, 7.0)]

        def fake_step(_state, batch, _step, **_kwargs):
            loss = parameter * batch["value"]
            zero = loss * 0
            return DiffusionStepOutput(
                loss=loss, loss_diff=loss, loss_teacher=zero, loss_delta=zero,
                loss_anchor=zero, loss_prefix=zero, loss_norm=zero,
                clip_scaffold_scale=1.0)

        with patch("train.make_streaming_dataloader", return_value=iter(batches)), \
             patch("train.diffusion_training_step", side_effect=fake_step):
            summary = run_diffusion_training_loop(
                state,
                DiffusionPhaseSpec("test", "test", "test", 1, 1, 4, 2,
                                   0.0, False, 0.0),
                optimizer, [parameter])

        self.assertEqual(summary["steps"], 2)
        # The second update averages 0.8 * 5 and 0.8 * 7, not just the
        # final microbatch's 5.6 loss.
        self.assertAlmostEqual(summary["final_loss"], 4.8)
        self.assertAlmostEqual(parameter.item(), 0.2)

    def test_residual_drift_uses_a_window_and_aborts(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        cfg = TrainConfig(
            connector_type="clip_gemma_residual_tsc",
            run_clip_alignment_pretrain=False,
            conditioning_dropout_prob=0.0,
            gradient_accumulation_steps=1, validation_every_opt_steps=0,
            generation_grid_every_opt_steps=0, max_samples_ella=50,
            ella_epochs=1, ella_max_opt_steps=50, train_batch_size=1,
            residual_drift_warn_ratio=0.1,
            residual_drift_stop_ratio=0.2,
            grad_clip_norm=100.0,
        )
        state = SimpleNamespace(cfg=cfg, device=torch.device("cpu"), wandb=None)
        batches = [{"ratio": 0.0}] * 25 + [{"ratio": 0.3}] * 25

        def fake_step(_state, batch, _step, **_kwargs):
            loss = parameter * 0
            return DiffusionStepOutput(
                loss=loss, loss_diff=loss, loss_teacher=loss, loss_delta=loss,
                loss_anchor=loss, loss_prefix=loss, loss_norm=loss,
                clip_scaffold_scale=1.0,
                residual_norm_ratio=torch.tensor([batch["ratio"]]),
                timesteps=torch.tensor([500]),
            )

        with patch("train.make_streaming_dataloader", return_value=iter(batches)), \
             patch("train.diffusion_training_step", side_effect=fake_step):
            # A cumulative mean would be 0.15 here and miss the 0.2 gate.
            # The second 25-step window is 0.3 and must abort.
            with self.assertRaisesRegex(RuntimeError, "Residual drift stop"):
                run_diffusion_training_loop(
                    state,
                    DiffusionPhaseSpec("test", "test", "test", 1, 1, 50, 50,
                                       0.0, False, 0.0),
                    optimizer, [parameter])

    def test_residual_checkpoint_schema_pins_clip_basis_and_scale(self):
        cfg = TrainConfig(
            connector_type="clip_gemma_residual_tsc",
            run_clip_alignment_pretrain=False,
            conditioning_dropout_prob=0.0,
            sd_checkpoint="/tmp/stylejourney.safetensors",
            use_sd_checkpoint_text_encoder=True,
            residual_strength=0.75,
        )
        state = SimpleNamespace(cfg=cfg, gemma_hidden_size=1152)
        checkpoint = {"connector_schema": connector_checkpoint_schema(state)}
        validate_connector_checkpoint_schema(checkpoint, state, "test.pt")

        changed_basis = SimpleNamespace(
            cfg=TrainConfig(
                connector_type="clip_gemma_residual_tsc",
                run_clip_alignment_pretrain=False,
                conditioning_dropout_prob=0.0,
                sd_checkpoint="/tmp/stylejourney.safetensors",
                use_sd_checkpoint_text_encoder=False,
                residual_strength=0.75,
            ),
            gemma_hidden_size=1152,
        )
        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            validate_connector_checkpoint_schema(checkpoint, changed_basis, "test.pt")

        changed_scale = SimpleNamespace(
            cfg=TrainConfig(
                connector_type="clip_gemma_residual_tsc",
                run_clip_alignment_pretrain=False,
                conditioning_dropout_prob=0.0,
                sd_checkpoint="/tmp/stylejourney.safetensors",
                use_sd_checkpoint_text_encoder=True,
                residual_strength=1.0,
            ),
            gemma_hidden_size=1152,
        )
        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            validate_connector_checkpoint_schema(checkpoint, changed_scale, "test.pt")

    def test_schema_v1_pure_connector_checkpoint_remains_loadable(self):
        cfg = TrainConfig(connector_type="ella_tsc")
        state = SimpleNamespace(cfg=cfg, gemma_hidden_size=1152)
        legacy_schema = connector_checkpoint_schema(state)
        legacy_schema["version"] = 1
        for key in (
            "use_sd_checkpoint_text_encoder", "sd_checkpoint",
            "residual_strength",
        ):
            legacy_schema.pop(key)
        validate_connector_checkpoint_schema(
            {"connector_schema": legacy_schema}, state, "pure-v1.pt")

    def test_default_sd_checkpoint_schema_is_cwd_independent(self):
        cfg = TrainConfig(connector_type="ella_tsc")
        schema = connector_checkpoint_schema(
            SimpleNamespace(cfg=cfg, gemma_hidden_size=1152))
        self.assertEqual(schema["sd_checkpoint"], "")

    def test_clip_checkpoint_namespace_and_buffer_filters(self):
        transformers4_keys = {"text_model.embeddings.position_embedding.weight"}
        transformers5_keys = {"embeddings.position_embedding.weight"}
        self.assertEqual(
            compatible_clip_checkpoint_key(
                "embeddings.position_embedding.weight", transformers4_keys),
            "text_model.embeddings.position_embedding.weight",
        )
        self.assertEqual(
            compatible_clip_checkpoint_key(
                "text_model.embeddings.position_embedding.weight",
                transformers5_keys),
            "embeddings.position_embedding.weight",
        )
        missing, unexpected = filter_clip_checkpoint_incompatibilities(
            ["text_model.embeddings.position_ids", "encoder.weight"],
            ["embeddings.position_ids", "text_projection.weight", "bad.weight"],
        )
        self.assertEqual(missing, ["encoder.weight"])
        self.assertEqual(unexpected, ["bad.weight"])

    def test_residual_sara_requires_p1_resume(self):
        cfg = TrainConfig(
            experiment_stage="stage3_long_context_with_sara",
            connector_type="clip_gemma_residual_tsc",
            run_clip_alignment_pretrain=False,
            conditioning_dropout_prob=0.0,
        )
        with self.assertRaisesRegex(ValueError, "requires --resume-ckpt"):
            validate_residual_sara_resume(cfg, {"sara"}, "")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "p1.pt"
            torch.save({
                "stage": "ella_frozen_unet",
                "completion": {
                    "completed": True, "phase": "ella",
                    "optimizer_steps": 1,
                },
            }, checkpoint_path)
            validate_residual_sara_resume(
                cfg, {"sara"}, str(checkpoint_path))

            torch.save({
                "stage": "clip_gemma_residual_frozen_unet",
                "completion": {
                    "completed": True, "phase": "ella",
                    "optimizer_steps": 1,
                },
            }, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "intermediate"):
                validate_residual_sara_resume(
                    cfg, {"sara"}, str(checkpoint_path))

            torch.save({
                "stage": "ella_frozen_unet",
                "completion": {
                    "completed": True, "phase": "ella",
                    "optimizer_steps": 0,
                },
            }, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "intermediate"):
                validate_residual_sara_resume(
                    cfg, {"sara"}, str(checkpoint_path))

        disabled_cfg = TrainConfig(
            experiment_stage="stage3_long_context_with_sara",
            connector_type="clip_gemma_residual_tsc",
            run_clip_alignment_pretrain=False,
            conditioning_dropout_prob=0.0,
            sara_epochs=0,
        )
        validate_residual_sara_resume(disabled_cfg, {"sara"}, "")

    def test_paired_evaluator_uses_exact_sign_test_and_content_mask(self):
        p_value, wins, losses, non_ties = exact_sign_test_two_sided(
            torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]).numpy())
        self.assertEqual((wins, losses, non_ties), (4, 0, 4))
        self.assertAlmostEqual(p_value, 0.125)
        prediction = torch.tensor([[[[1.0, 100.0]]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[[1.0, 0.0]]]])
        self.assertTrue(torch.equal(
            masked_per_sample_mse(prediction, target, mask),
            torch.tensor([1.0])))

    def test_shared_diffusion_step_backpropagates_to_connector(self):
        class LatentDistribution:
            def __init__(self, latent):
                self.latent = latent

            def sample(self):
                return self.latent

        class VAE(nn.Module):
            config = SimpleNamespace(scaling_factor=1.0)

            def encode(self, image):
                extra = torch.zeros_like(image[:, :1])
                latent = torch.cat([image, extra], dim=1)
                return SimpleNamespace(
                    latent_dist=LatentDistribution(latent)
                )

        class Connector(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.25))

            def forward(self, hidden, _timestep, _mask, context_tokens):
                pooled = hidden.mean(dim=1, keepdim=True) * self.scale
                return pooled.expand(-1, context_tokens, -1)

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.class_embedding = None

            def forward(self, noisy, _timestep, encoder_hidden_states,
                        class_labels=None):
                influence = encoder_hidden_states.mean(dim=(1, 2))
                influence = influence.reshape(-1, 1, 1, 1)
                if self.class_embedding is not None:
                    influence = influence + self.class_embedding(
                        class_labels).reshape(-1, 1, 1, 1)
                return SimpleNamespace(sample=noisy + influence)

        class Scheduler:
            config = SimpleNamespace(num_train_timesteps=1000)

            @staticmethod
            def add_noise(latent, noise, _timestep):
                return latent + noise

        cfg = TrainConfig(
            context_tokens=4,
            clip_anchor_tokens=4,
            max_gemma_len=8,
            conditioning_dropout_prob=0.0,
            caption_mix_short=0.0,
            caption_mix_medium=0.0,
            caption_mix_long=1.0,
            lambda_teacher=0.0,
            lambda_text_delta=0.0,
            mixed_precision="no",
        )
        connector = Connector()
        state = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            unet_dtype=torch.float32,
            autocast_dtype=None,
            vae=VAE(),
            connector=connector,
            unet=UNet(),
            scheduler=Scheduler(),
            clip_model=None,
            caption_availability_logged=True,
            encode_gemma=lambda captions: (
                torch.ones(len(captions), 3, 4),
                torch.ones(len(captions), 3, dtype=torch.long),
            ),
        )
        batch = {
            "image": torch.zeros(1, 3, 8, 8),
            "image_mask": torch.ones(1, 1, 8, 8),
            "caption": ["a long caption"],
        }

        output = diffusion_training_step(
            state,
            batch,
            0,
            use_teacher_delta=False,
            semantic_anchor_weight=0.0,
        )
        output.loss.backward()

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.loss_teacher.item(), 0.0)
        self.assertEqual(output.loss_delta.item(), 0.0)
        self.assertIsNotNone(connector.scale.grad)
        self.assertNotEqual(connector.scale.grad.item(), 0.0)

        cfg.camera_conditioning_enabled = True
        state.camera_metadata_logged = True
        state.camera_metadata_samples = 0
        state.camera_presence_counts = [0, 0, 0, 0]
        conditioner = CameraConditioner(
            output_dim=1, hidden_dim=8, fourier_bands=2,
            capture_embed_dim=4, enable_raw_focal_head=True)
        install_camera_conditioner(state.unet, conditioner)
        batch["camera_condition"] = make_camera_condition(
            focal_length_mm=35, capture_type="photo").unsqueeze(0)
        output = diffusion_training_step(
            state, batch, 1, use_teacher_delta=False,
            semantic_anchor_weight=0.0)
        output.loss.backward()

        self.assertGreater(
            torch.count_nonzero(
                conditioner.raw_focal_head.mlp[-1].weight.grad).item(), 0)
        self.assertEqual(state.camera_metadata_samples, 1)
        self.assertEqual(state.camera_presence_counts, [0, 1, 0, 0])

    def test_residual_short_prompt_bypass_is_exact_and_skips_gemma(self):
        class Tokenizer:
            def __call__(self, prompts, **_kwargs):
                return {"input_ids": [[1, 2, 3] for _ in prompts]}

        cfg = SimpleNamespace(
            clip_anchor_tokens=4, context_tokens=4, residual_strength=1.0)
        clip_context = torch.randn(2, 4, 768)
        state = SimpleNamespace(
            cfg=cfg, device=torch.device("cpu"), unet_dtype=torch.float32,
            clip_tokenizer=Tokenizer(),
            encode_clip=lambda _prompts: (clip_context, None),
            encode_gemma=lambda _prompts: self.fail("Gemma must not run"),
            connector=lambda *_args, **_kwargs: self.fail("connector must not run"),
        )
        context, full, prefix, long_mask = residual_context_for_captions(
            state, ["short", "also short"], torch.tensor([2, 3]))
        self.assertTrue(torch.equal(context, clip_context))
        self.assertFalse(long_mask.any())
        self.assertEqual(full.item(), 0.0)
        self.assertEqual(prefix.item(), 0.0)

    def test_residual_connector_starts_as_zero_delta(self):
        connector = build_connector(
            "clip_gemma_residual_tsc", gemma_dim=16, width=32,
            context_tokens=77, anchor_tokens=77, layers=2, heads=4,
            ff_mult=2, time_embed_dim=32, gemma_layer_mix_count=4)
        delta = connector(
            torch.randn(2, 77, 768), torch.randn(2, 4, 12, 16),
            torch.tensor([10, 500]), torch.ones(2, 12, dtype=torch.long))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_tsc_reads_long_input_and_preserves_sd_token_contract(self):
        connector = build_connector(
            "ella_tsc",
            gemma_dim=16,
            width=32,
            context_tokens=77,
            anchor_tokens=77,
            layers=2,
            heads=4,
            ff_mult=2,
            dropout=0.0,
            time_embed_dim=32,
            extra_gate_init=-5.0,
            gemma_layer_mix_count=4,
        )
        gemma_states = torch.randn(2, 4, 256, 16)
        mask = torch.ones(2, 256, dtype=torch.long)
        output = connector(gemma_states, torch.tensor([1, 500]), mask)

        self.assertEqual(output.shape, (2, 77, 32))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaisesRegex(ValueError, "fixed output length"):
            connector(gemma_states, torch.tensor([1, 500]), mask,
                      context_tokens=256)

    def test_full_frame_resize_keeps_the_entire_image(self):
        image = Image.new("RGB", (1200, 400), color=(255, 0, 0))
        resized, content_mask = resize_full_frame(image, (640, 384))

        self.assertEqual(resized.size, (640, 384))
        self.assertEqual(content_mask.getbbox(), (0, 85, 640, 298))
        self.assertEqual(resized.getpixel((320, 192)), (255, 0, 0))
        self.assertEqual(resized.getpixel((320, 0)), (127, 127, 127))

    def test_long_edge_resize_preserves_aspect_ratio(self):
        image = Image.new("RGB", (4000, 2000), color=(1, 2, 3))

        resized = resize_long_edge(image, 1024)

        self.assertEqual(resized.size, (1024, 512))

    def test_unsplash_schema_loads_local_image_without_cropping(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            Image.new("RGB", (400, 200), color=(10, 20, 30)).save(image_path)
            source = [{
                "local_image_path": str(image_path),
                "training_caption": "a detailed landscape caption",
                "caption_short": None,
            }]

            sample = next(iter(StreamingSDDataset(
                source, max_samples=1, buckets=[(64, 32)],
                max_image_dimension=1024)))

        self.assertEqual(sample["caption"], "a detailed landscape caption")
        self.assertEqual(sample["caption_short"], "")
        self.assertEqual(tuple(sample["image"].shape), (3, 32, 64))
        self.assertTrue(sample["image_mask"].bool().all())

    def test_multiple_sources_are_consumed_without_schema_alignment(self):
        sources = [
            (iter([{"id": "a1"}, {"id": "a2"}]), "/a"),
            (iter([{"other": "b1"}]), "/b"),
        ]

        samples = list(RoundRobinSources(sources, seed=1))

        self.assertEqual(len(samples), 3)
        self.assertEqual({sample["_source_root"] for sample in samples}, {"/a", "/b"})
        self.assertEqual({sample.get("id") for sample in samples if "id" in sample},
                         {"a1", "a2"})

    def test_local_data_sources_accept_parquet_files_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.parquet"
            second = root / "nested" / "second.parquet"
            second.parent.mkdir()
            first.touch()
            second.touch()

            self.assertEqual(_local_parquet_files(str(first)), [str(first.resolve())])
            windows_style = str(first).replace("/", "\\")
            self.assertEqual(
                _local_parquet_files(windows_style), [str(first.resolve())])
            self.assertEqual(
                _local_parquet_files(str(root)),
                [str(first.resolve()), str(second.resolve())],
            )

    def test_bucket_batcher_never_mixes_image_shapes(self):
        samples = [
            {
                "image": torch.zeros(3, 512, 512),
                "image_mask": torch.ones(1, 512, 512),
                "caption": "square", "bucket": (512, 512),
            },
            {
                "image": torch.zeros(3, 384, 640),
                "image_mask": torch.ones(1, 384, 640),
                "caption": "wide", "bucket": (640, 384),
            },
            {
                "image": torch.zeros(3, 512, 512),
                "image_mask": torch.ones(1, 512, 512),
                "caption": "square two", "bucket": (512, 512),
            },
        ]
        batches = list(BucketBatchDataset(samples, batch_size=2))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["bucket"], (512, 512))
        self.assertEqual(tuple(batches[0]["image"].shape), (2, 3, 512, 512))

        batches_with_tail = list(BucketBatchDataset(
            samples, batch_size=2, drop_last=False))
        self.assertEqual(batches_with_tail[1]["bucket"], (640, 384))
        self.assertEqual(
            tuple(batches_with_tail[1]["image"].shape), (1, 3, 384, 640))

    def test_config_rejects_unknown_keys_and_unsafe_output_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"unknown_key": 1}))
            with self.assertRaisesRegex(ValueError, "Unknown config keys"):
                TrainConfig.from_json(str(path))

        with self.assertRaisesRegex(ValueError, "77-token conditioning"):
            TrainConfig(context_tokens=256, clip_anchor_tokens=77)

    def test_missing_local_checkpoint_does_not_become_a_hf_model_id(self):
        cfg = TrainConfig(sd_checkpoint="/missing/model.safetensors")
        with patch("pure_ella.config.os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                resolve_sd_checkpoint(cfg)

    def test_suffix_diagnostic_proves_late_non_truncated_input(self):
        class WhitespaceTokenizer:
            def __init__(self):
                self.calls = 0

            def encode(self, text, **_kwargs):
                self.calls += 1
                return [0] + text.split()

        case = {
            "name": "late_object",
            "prompt_suffix_a": "make the object red",
            "prompt_suffix_b": "make the object blue",
        }
        cfg = SimpleNamespace(
            suffix_counterfactual_prefix=" ".join(["prefix"] * 80),
            suffix_counterfactual_cases=[case],
            clip_anchor_tokens=77,
            max_gemma_len=100,
        )
        tokenizer = WhitespaceTokenizer()
        state = SimpleNamespace(cfg=cfg, gemma_tokenizer=tokenizer)

        result = validate_suffix_counterfactual_token_boundaries(state)
        self.assertGreater(result["late_object"]["prefix_tokens"], 77)
        self.assertEqual(tokenizer.calls, 3)
        self.assertIs(
            validate_suffix_counterfactual_token_boundaries(state), result)
        self.assertEqual(tokenizer.calls, 3)

        cfg.max_gemma_len = 82
        with self.assertRaisesRegex(ValueError, "max_gemma_len"):
            validate_suffix_counterfactual_token_boundaries(state)

    def test_gemma_encoding_tokenizes_once_and_reuses_ids_for_padding(self):
        class Tokenizer:
            truncation_side = "right"

            def __init__(self):
                self.calls = 0

            def __call__(self, prompts, **kwargs):
                self.calls += 1
                self.assert_untruncated(kwargs)
                input_ids = [list(range(1, len(text.split()) + 2))
                             for text in prompts]
                return {
                    "input_ids": input_ids,
                    "attention_mask": [[1] * len(ids) for ids in input_ids],
                }

            @staticmethod
            def assert_untruncated(kwargs):
                if kwargs != {"padding": False, "truncation": False}:
                    raise AssertionError(kwargs)

            @staticmethod
            def pad(encoded, padding, max_length, return_tensors):
                if padding not in {"max_length", "longest"} or return_tensors != "pt":
                    raise AssertionError((padding, return_tensors))
                target_length = (
                    max_length if padding == "max_length"
                    else max(len(sequence) for sequence in encoded["input_ids"])
                )
                padded = {}
                for name, sequences in encoded.items():
                    pad_value = 0
                    padded[name] = torch.tensor([
                        sequence + [pad_value] * (target_length - len(sequence))
                        for sequence in sequences
                    ])
                return BatchEncoding(padded)

        class Model:
            device = torch.device("cpu")

            def __call__(self, input_ids, **_kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
                return SimpleNamespace(hidden_states=(hidden, hidden + 1))

        cfg = TrainConfig(
            context_tokens=4,
            clip_anchor_tokens=4,
            max_gemma_len=6,
            gemma_layer_index=-1,
            gemma_layer_mix_count=2,
        )
        tokenizer = Tokenizer()
        state = SimpleNamespace(
            cfg=cfg,
            gemma_tokenizer=tokenizer,
            gemma_model=Model(),
            device=torch.device("cpu"),
            gemma_prompt_max_observed=-1,
            gemma_truncated_prompt_count=0,
        )

        hidden, mask = make_encode_gemma(state)(["one two", "three"])

        self.assertEqual(tokenizer.calls, 1)
        self.assertEqual(hidden.shape, (2, 2, 6, 3))
        self.assertEqual(mask.shape, (2, 6))

        hidden, mask = make_encode_gemma(state)(
            ["one two", "three"], pad_to_max_length=False)
        self.assertEqual(hidden.shape, (2, 2, 3, 3))
        self.assertEqual(mask.shape, (2, 3))

        long_prompt = "one two three four five six seven"
        with self.assertRaisesRegex(ValueError, "refusing silent truncation"):
            make_encode_gemma(state)([long_prompt])
        self.assertEqual(tokenizer.calls, 3)
        self.assertEqual(state.gemma_truncated_prompt_count, 0)

        cfg.fail_on_prompt_truncation = False
        hidden, mask = make_encode_gemma(state)([long_prompt])
        self.assertEqual(tokenizer.calls, 4)
        self.assertEqual(hidden.shape, (1, 2, 6, 3))
        self.assertEqual(mask.sum().item(), 6)
        self.assertEqual(state.gemma_truncated_prompt_count, 1)

    def test_caption_curriculum_uses_nearest_available_fallback(self):
        cfg = TrainConfig(
            caption_mix_short=0.25,
            caption_mix_medium=0.25,
            caption_mix_long=0.5,
        )
        state = SimpleNamespace(cfg=cfg, caption_availability_logged=True)
        batch = {
            "caption": ["long one", "long two"],
            "caption_medium": ["medium one", ""],
            "caption_short": ["", "short two"],
        }

        with patch("train.torch.rand", return_value=torch.tensor([0.1, 0.3])):
            selected = select_training_captions(batch, state)

        self.assertEqual(selected, ["medium one", "short two"])

    def test_suffix_diagnostic_runs_for_long_input_with_fixed_output(self):
        class WhitespaceTokenizer:
            @staticmethod
            def encode(text, **_kwargs):
                return [0] + text.split()

        class Connector(nn.Module):
            def forward(self, hidden, _timestep, _mask, context_tokens):
                pooled = hidden.mean(dim=1, keepdim=True)
                return pooled.expand(-1, context_tokens, -1)

        class UNet(nn.Module):
            def forward(self, noisy, _timestep, encoder_hidden_states):
                influence = encoder_hidden_states.mean().reshape(1, 1, 1, 1)
                return SimpleNamespace(sample=noisy + influence)

        class Scheduler:
            @staticmethod
            def add_noise(latent, noise, _timestep):
                return latent + noise

        prefix = " ".join(["prefix"] * 80)
        case = {
            "name": "late_object",
            "prompt_suffix_a": "make the object red",
            "prompt_suffix_b": "make the object blue blue",
            "seed": 1,
        }
        cfg = SimpleNamespace(
            context_tokens=77,
            clip_anchor_tokens=77,
            max_gemma_len=100,
            suffix_counterfactual_prefix=prefix,
            suffix_counterfactual_cases=[case],
            suffix_diagnostic_timestep=500,
        )

        def encode_gemma(prompts):
            values = torch.tensor([
                float(len(prompt.split())) for prompt in prompts
            ]).reshape(-1, 1, 1)
            return values.expand(-1, 4, 8), torch.ones(
                len(prompts), 4, dtype=torch.long)

        connector = Connector().train()
        unet = UNet().train()
        state = SimpleNamespace(
            cfg=cfg,
            gemma_tokenizer=WhitespaceTokenizer(),
            connector=connector,
            unet=unet,
            scheduler=Scheduler(),
            device=torch.device("cpu"),
            unet_dtype=torch.float32,
            encode_gemma=encode_gemma,
        )

        result = suffix_counterfactual_sensitivity(state)

        self.assertGreater(result["suffix_sensitivity_mean"], 0.0)
        self.assertTrue(connector.training)
        self.assertTrue(unet.training)

    def test_trm_uses_configured_layer_mixture(self):
        connector = build_connector(
            "trm_yz",
            gemma_dim=16,
            width=32,
            context_tokens=77,
            anchor_tokens=77,
            layers=1,
            heads=4,
            ff_mult=2,
            dropout=0.0,
            time_embed_dim=32,
            extra_gate_init=-5.0,
            gemma_layer_mix_count=4,
            trm_outer_steps=1,
            trm_inner_steps=1,
            trm_scratch_tokens=4,
            trm_y_gate_init=-2.0,
            trm_z_gate_init=-1.0,
        )
        states = torch.randn(1, 4, 20, 16)
        output = connector(
            states, torch.tensor([100]), torch.ones(1, 20, dtype=torch.long))
        output.square().mean().backward()

        self.assertEqual(output.shape, (1, 77, 32))
        self.assertIsNotNone(connector.layer_mix_logits.grad)

    def test_sara_hook_keeps_boolean_mask(self):
        module = nn.Linear(4, 2, bias=False)
        module.weight._sara_sparse_mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True]])
        handles = install_sara_gradient_masks(module)
        hook = next(iter(module.weight._backward_hooks.values()))
        captured_tensors = [
            cell.cell_contents for cell in hook.__closure__ or ()
            if isinstance(cell.cell_contents, torch.Tensor)
        ]
        module.weight.sum().backward()

        self.assertTrue(any(tensor.dtype == torch.bool for tensor in captured_tensors))
        self.assertEqual(module.weight.grad.count_nonzero().item(), 4)
        remove_sara_gradient_masks(handles)


if __name__ == "__main__":
    unittest.main()

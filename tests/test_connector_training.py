import json
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
    diffusion_training_step,
    estimate_steps,
    make_encode_gemma,
    resolve_autocast_dtype,
    resolve_model_weight_dtype,
    select_training_captions,
)
from pure_ella.config import TrainConfig, resolve_sd_checkpoint
from pure_ella.connector import build_connector
from pure_ella.dataset import (
    BucketBatchDataset,
    RoundRobinSources,
    StreamingSDDataset,
    _local_parquet_files,
    resize_full_frame,
    resize_long_edge,
)
from pure_ella.diagnostics import (
    suffix_counterfactual_sensitivity,
    validate_suffix_counterfactual_token_boundaries,
)
from pure_ella.sara import install_sara_gradient_masks, remove_sara_gradient_masks


class ConnectorTrainingTests(unittest.TestCase):
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
        with self.assertRaisesRegex(ValueError, "SaRA requires"):
            TrainConfig(
                experiment_stage="stage3_long_context_with_sara",
                model_weight_dtype="bfloat16",
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
            def forward(self, noisy, _timestep, encoder_hidden_states):
                influence = encoder_hidden_states.mean(dim=(1, 2))
                influence = influence.reshape(-1, 1, 1, 1)
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
                if padding != "max_length" or return_tensors != "pt":
                    raise AssertionError((padding, return_tensors))
                padded = {}
                for name, sequences in encoded.items():
                    pad_value = 0
                    padded[name] = torch.tensor([
                        sequence + [pad_value] * (max_length - len(sequence))
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

        long_prompt = "one two three four five six seven"
        with self.assertRaisesRegex(ValueError, "refusing silent truncation"):
            make_encode_gemma(state)([long_prompt])
        self.assertEqual(tokenizer.calls, 2)
        self.assertEqual(state.gemma_truncated_prompt_count, 0)

        cfg.fail_on_prompt_truncation = False
        hidden, mask = make_encode_gemma(state)([long_prompt])
        self.assertEqual(tokenizer.calls, 3)
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

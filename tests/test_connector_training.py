import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from pure_ella.config import TrainConfig, resolve_sd_checkpoint
from pure_ella.connector import build_connector
from pure_ella.dataset import BucketBatchDataset, resize_full_frame
from pure_ella.diagnostics import (
    suffix_counterfactual_sensitivity,
    validate_suffix_counterfactual_token_boundaries,
)
from pure_ella.sara import install_sara_gradient_masks, remove_sara_gradient_masks


class ConnectorTrainingTests(unittest.TestCase):
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
            @staticmethod
            def encode(text, **_kwargs):
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
        state = SimpleNamespace(cfg=cfg, gemma_tokenizer=WhitespaceTokenizer())

        result = validate_suffix_counterfactual_token_boundaries(state)
        self.assertGreater(result["late_object"]["prefix_tokens"], 77)

        cfg.max_gemma_len = 82
        with self.assertRaisesRegex(ValueError, "max_gemma_len"):
            validate_suffix_counterfactual_token_boundaries(state)

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

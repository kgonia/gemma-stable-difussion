import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from pure_ella.config import TrainConfig, resolve_sd_checkpoint
from pure_ella.connector import build_connector
from pure_ella.dataset import BucketBatchDataset, resize_full_frame


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

        self.assertEqual(batches[0]["bucket"], (512, 512))
        self.assertEqual(tuple(batches[0]["image"].shape), (2, 3, 512, 512))
        self.assertEqual(batches[1]["bucket"], (640, 384))
        self.assertEqual(tuple(batches[1]["image"].shape), (1, 3, 384, 640))

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


if __name__ == "__main__":
    unittest.main()

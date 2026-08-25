from __future__ import annotations

import tempfile
import unittest

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from scripts.visualization.validation_visualizer import log_validation_images


class TestValidationVisualizer(unittest.TestCase):
    def test_probability_heatmap_is_written_as_tensorboard_image(self) -> None:
        image = np.full((6, 8, 3), 64, dtype=np.uint8)
        probability_map = np.linspace(
            0.0,
            1.0,
            num=image.shape[0] * image.shape[1],
            dtype=np.float32,
        ).reshape(image.shape[:2])
        sample = {
            "image": image,
            "gt_points": np.asarray([[2.0, 3.0]], dtype=np.float32),
            "predictions": {
                "logits": np.asarray([0.9], dtype=np.float32),
                "points": np.asarray([[2.0, 3.0]], dtype=np.float32),
                "expert_indices": np.asarray([0], dtype=np.int64),
            },
            "image_path": "sample.jpg",
            "pred_count": 1.0,
            "prob_map": probability_map,
        }

        with tempfile.TemporaryDirectory() as log_dir:
            writer = SummaryWriter(log_dir=log_dir)
            log_validation_images(
                writer,
                "val_images/demo",
                [sample],
                epoch=7,
            )
            writer.close()

            event_accumulator = EventAccumulator(log_dir)
            event_accumulator.Reload()

        image_tags = event_accumulator.Tags()["images"]
        heatmap_tag = "val_images/demo/probability_heatmap_00"
        self.assertIn("val_images/demo/sample_00", image_tags)
        self.assertIn(heatmap_tag, image_tags)

        heatmap_events = event_accumulator.Images(heatmap_tag)
        self.assertEqual(len(heatmap_events), 1)
        self.assertEqual(heatmap_events[0].step, 7)
        self.assertEqual(heatmap_events[0].width, image.shape[1])
        self.assertEqual(heatmap_events[0].height, image.shape[0])


if __name__ == "__main__":
    unittest.main()

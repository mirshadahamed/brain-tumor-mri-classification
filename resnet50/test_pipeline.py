"""Quick local checks that do not access the held-out test images."""

import unittest
import math

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from common import PadToSquare, configure_phase, evaluation_transform, image_folder, make_model, train_transform
from spec import CLASS_TO_IDX, DEFAULT_DATA_ROOT
from train import run_epoch


class ResNet50PipelineTests(unittest.TestCase):
    def test_fixed_train_and_validation_mapping(self):
        for split, expected_count in (("train", 4673), ("validation", 1003)):
            dataset = image_folder(DEFAULT_DATA_ROOT, split, transform=None)
            self.assertEqual(dataset.class_to_idx, CLASS_TO_IDX)
            self.assertEqual(len(dataset), expected_count)

    def test_padding_preserves_the_entire_image(self):
        image = Image.new("RGB", (5, 3), "red")
        padded = PadToSquare()(image)
        self.assertEqual(padded.size, (5, 5))
        self.assertEqual(padded.getpixel((0, 1)), (255, 0, 0))
        self.assertEqual(padded.getpixel((4, 3)), (255, 0, 0))
        self.assertEqual(padded.getpixel((0, 0)), (0, 0, 0))

    def test_transforms_produce_resnet_input(self):
        image = Image.new("RGB", (300, 200), "gray")
        self.assertEqual(tuple(train_transform()(image).shape), (3, 224, 224))
        first = evaluation_transform()(image)
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertTrue(torch.equal(first, evaluation_transform()(image)))

    def test_classifier_outputs_four_logits(self):
        model = make_model(pretrained=False).eval()
        with torch.inference_mode():
            result = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(result.shape), (1, 4))

    def test_one_cpu_training_step(self):
        dataset = image_folder(DEFAULT_DATA_ROOT, "train", train_transform())
        sample = Subset(dataset, [0, 1244, 2283, 3443])
        loader = DataLoader(sample, batch_size=4, shuffle=False)
        model = make_model(pretrained=False)
        configure_phase(model, "head")
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scores = run_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, scaler, "head")
        self.assertTrue(math.isfinite(scores["loss"]))
        self.assertGreaterEqual(scores["accuracy"], 0)
        self.assertLessEqual(scores["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()

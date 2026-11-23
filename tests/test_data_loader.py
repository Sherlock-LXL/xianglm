from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import torch

from src.config import SOURCE_WEIGHTS
from src.data_loader import MixedPretrainingLoader
from tests.support import create_tokenized_fixture


class MixedPretrainingLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_directory = tempfile.TemporaryDirectory()
        cls.data_config = create_tokenized_fixture(Path(cls.data_directory.name), SOURCE_WEIGHTS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.data_directory.cleanup()

    def setUp(self) -> None:
        self.loader = MixedPretrainingLoader(
            data_config=self.data_config,
            context_length=32,
            split=self.data_config.train_split,
            seed=1234,
            max_open_files=4,
        )

    def tearDown(self) -> None:
        self.loader.close()

    def test_batch_shape_and_shift(self) -> None:
        batch = self.loader.next_batch(8)
        self.assertEqual(tuple(batch.input_ids.shape), (8, 32))
        self.assertEqual(tuple(batch.labels.shape), (8, 32))
        self.assertEqual(batch.input_ids.dtype, torch.int64)
        self.assertTrue(torch.equal(batch.input_ids[:, 1:], batch.labels[:, :-1]))
        self.assertGreaterEqual(int(batch.input_ids.min()), 0)
        self.assertLess(int(batch.input_ids.max()), self.loader.vocab_size)

    def test_state_restores_exact_batch(self) -> None:
        state = self.loader.state_dict()
        first = self.loader.next_batch(4)
        self.loader.load_state_dict(state)
        second = self.loader.next_batch(4)
        self.assertTrue(torch.equal(first.input_ids, second.input_ids))
        self.assertEqual(first.source_counts, second.source_counts)

    def test_source_mixture(self) -> None:
        counts = {name: 0 for name, _ in self.data_config.source_weights}
        for _ in range(100):
            batch = self.loader.next_batch(16)
            for name, count in batch.source_counts.items():
                counts[name] += count
        total = sum(counts.values())
        expected = self.data_config.source_weight_dict
        for name, target in expected.items():
            observed = counts[name] / total
            self.assertLess(
                abs(observed - target), 0.06, msg=(f"{name}: observed={observed}, target={target}")
            )


if __name__ == "__main__":
    unittest.main()

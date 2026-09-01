from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import torch

from src.checkpoint import BEST_CHECKPOINT_NAME, capture_rng_state, load_training_checkpoint, restore_rng_state, save_training_checkpoint
from src.config import SOURCE_WEIGHTS, ExperimentConfig, ModelConfig, TrainConfig
from src.data_loader import MixedPretrainingLoader
from src.model import TransformerLM
from tests.support import create_tokenized_fixture

class CheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_directory = tempfile.TemporaryDirectory()
        data_config = create_tokenized_fixture(Path(self.data_directory.name), SOURCE_WEIGHTS)
        self.experiment = ExperimentConfig(
            name="checkpoint_unit_test",
            description="checkpoint test",
            model=ModelConfig(vocab_size=128, context_length=16, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, tie_embeddings=True, use_bias=False),
            train=TrainConfig(target_tokens=10_000, micro_batch_size=2, gradient_accumulation_steps=1, max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_tokens=1_000),
            data=data_config,
        )
        self.model = TransformerLM(self.experiment.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loader = MixedPretrainingLoader(data_config=self.experiment.data, context_length=16, split=(self.experiment.data.train_split), seed=99)

    def tearDown(self) -> None:
        self.loader.close()
        self.data_directory.cleanup()

    def test_save_and_restore(self) -> None:
        batch = self.loader.next_batch(2)
        input_ids = batch.input_ids % 128
        labels = batch.labels % 128
        output = self.model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        expected_parameters = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}
        expected_loader_state = (self.loader.state_dict())
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint_path = (
                save_training_checkpoint(
                    directory=directory,
                    experiment=self.experiment,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=None,
                    train_loader=self.loader,
                    step=17,
                    tokens_processed=12_345,
                    best_validation_loss=3.25,
                    is_best=True,
                )
            )
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.add_(1.0)
            self.loader.next_batch(3)
            restored = (
                load_training_checkpoint(
                    path=checkpoint_path,
                    experiment=(self.experiment),
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=None,
                    train_loader=self.loader,
                    map_location="cpu",
                )
            )
            self.assertEqual(restored.step, 17)
            self.assertEqual(restored.tokens_processed, 12_345)
            self.assertEqual(restored.best_validation_loss, 3.25)
            for name, tensor in (self.model.state_dict().items()):
                self.assertTrue(torch.equal(tensor, expected_parameters[name]), msg=name)
            restored_loader_state = (self.loader.state_dict())
            self.assertEqual(restored_loader_state["rng_state"], expected_loader_state["rng_state"])
            self.assertTrue((directory / BEST_CHECKPOINT_NAME).exists())

    @unittest.skipUnless(torch.cuda.is_available(), "需要CUDA验证map_location场景")
    def test_rng_state_restores_after_cuda_mapping(self) -> None:
        state = capture_rng_state()
        state["torch_cpu"] = state["torch_cpu"].to("cuda")
        if state["torch_cuda"] is not None:
            state["torch_cuda"] = [generator_state.to("cuda") for generator_state in state["torch_cuda"]]
        restore_rng_state(state)

if __name__ == "__main__":
    unittest.main()
from __future__ import annotations
import unittest
import torch

from src.config import ModelConfig, estimate_parameter_count
from src.model import TransformerLM

class TransformerLMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_config = ModelConfig(vocab_size=128, context_length=32, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, tie_embeddings=True, use_bias=False)
        cls.model = TransformerLM(cls.model_config)

    def test_parameter_count(self) -> None:
        self.assertEqual(self.model.parameter_count(), estimate_parameter_count(self.model_config))

    def test_weight_tying(self) -> None:
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)

    def test_forward_and_backward(self) -> None:
        tokens = torch.randint(0, self.model_config.vocab_size, (2, 17))
        output = self.model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
        self.assertEqual(tuple(output.logits.shape), (2, 16, self.model_config.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        self.model.zero_grad(set_to_none=True)
        output.loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))

    def test_causal_attention(self) -> None:
        self.model.eval()
        torch.manual_seed(7)
        first = torch.randint(0, self.model_config.vocab_size, (1, 16))
        second = first.clone()
        second[:, 8:] = torch.randint(0, self.model_config.vocab_size, (1, 8))
        with torch.no_grad():
            first_logits = self.model(first).logits
            second_logits = self.model(second).logits
        self.assertTrue(torch.allclose(first_logits[:, :8], second_logits[:, :8], atol=1e-6, rtol=1e-5))

if __name__ == "__main__":
    unittest.main()
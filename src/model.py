from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from src.config import ModelConfig, estimate_parameter_count

@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.eps)
        normalized = normalized.to(dtype=input_dtype)
        return normalized * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_sequence_length: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE要求head_dim为偶数")
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        inverse_frequency = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embedding.cos(), persistent=False)
        self.register_buffer("sin_cached", embedding.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        first_half, second_half = x.chunk(2, dim=-1)
        return torch.cat((-second_half, first_half), dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        sequence_length = query.size(-2)
        if sequence_length > self.max_sequence_length:
            raise ValueError(f"输入序列超过RoPE缓存长度：{sequence_length} > {self.max_sequence_length}")
        cosine = self.cos_cached[:sequence_length].to(device=query.device, dtype=query.dtype).unsqueeze(0).unsqueeze(0)
        sine = self.sin_cached[:sequence_length].to(device=query.device, dtype=query.dtype).unsqueeze(0).unsqueeze(0)
        rotated_query = query * cosine + self.rotate_half(query) * sine
        rotated_key = key * cosine + self.rotate_half(key) * sine
        return rotated_query, rotated_key

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv_projection = nn.Linear(config.d_model, 3 * config.d_model, bias=config.use_bias)
        self.output_projection = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(head_dim=config.head_dim, max_sequence_length=config.context_length, theta=config.rope_theta)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        sequence_length = x.size(1)
        qkv = self.qkv_projection(x)
        qkv = qkv.view(batch_size, sequence_length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rope(query, key)
        attention_dropout = self.dropout if self.training else 0.0
        attention_output = (F.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=None, dropout_p=attention_dropout, is_causal=True))
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        output = self.output_projection(attention_output)
        return self.residual_dropout(output)

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.up_projection = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.down_projection = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_projection(x))
        up = self.up_projection(x)
        hidden = gate * up
        output = self.down_projection(hidden)
        return self.dropout(output)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(d_model=config.d_model, eps=config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(d_model=config.d_model, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig, *, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        config.validate()
        if config.use_bias:
            raise ValueError("当前三套预设和参数估算均采用无bias结构，请保持use_bias=False")
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(d_model=config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_module)
        self._initialize_residual_projections()
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._validate_parameter_count()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initialize_residual_projections(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output_projection.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down_projection.weight, mean=0.0, std=residual_std)

    def _validate_parameter_count(self) -> None:
        actual = self.parameter_count()
        expected = estimate_parameter_count(self.config)
        if actual != expected:
            raise RuntimeError(f"模型实际参数量与config估算不一致：\n实际：{actual:,}\n预计：{expected:,}")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids必须是二维张量：(batch, sequence)")
        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("输入序列不能为空")
        if sequence_length > self.config.context_length:
            raise ValueError(f"输入序列超过模型上下文长度：{sequence_length} > {self.config.context_length}")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            input_ids = input_ids.long()
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(f"labels形状必须与input_ids一致：{labels.shape} != {input_ids.shape}")
            if labels.dtype != torch.int64:
                labels = labels.long()
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = activation_checkpoint(block, x, use_reentrant=False, preserve_rng_state=self.config.dropout > 0.0)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100)
        return ModelOutput(logits=logits, loss=loss)
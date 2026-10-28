"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger

logger = get_logger("inference_engine")


class FullTransformerModel(nn.Module):
    """Fallback modular transformer wrapper combining embeddings, blocks, and lm_head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        if getattr(config, "tie_weights", False):
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[Any] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        h = self.embeddings(input_ids, offset=position_offset)
        h = self.blocks(h, kv_cache=kv_cache, **kwargs)
        logits = self.lm_head(h)
        return logits


class InferenceEngine:
    """
    End-to-end inference engine managing model state, token generation loop,
    and sampling orchestration.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        model: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        try:
            self.model_config = model_config or ModelConfig()
        except Exception:
            self.model_config = ModelConfig(d_model=768, vocab_size=50257)

        self.inference_config = inference_config or InferenceConfig()
        self.device = torch.device(device) if device else torch.device("cpu")

        if model is not None:
            self.model = model
        else:
            try:
                self.model = FullTransformerModel(self.model_config)
            except Exception as e:
                logger.debug(f"Could not build FullTransformerModel: {e}")
                self.model = None
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        if self.model is None:
            raise RuntimeError("Cannot generate tokens because no model is available")

        temp = temperature if temperature is not None else self.inference_config.temperature
        k = top_k if top_k is not None else self.inference_config.top_k
        p = top_p if top_p is not None else self.inference_config.top_p
        tokens = list(prompt_tokens)
        generator = torch.Generator(device=self.device)
        if self.inference_config.seed is not None:
            generator.manual_seed(self.inference_config.seed)
        use_cache = self.inference_config.use_kv_cache and isinstance(self.model, FullTransformerModel)
        if use_cache:
            from inference.generation import KVCache
            kv_cache = KVCache(num_layers=self.model_config.n_layers)

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                if use_cache:
                    past_length = kv_cache.get_seq_length()
                    model_tokens = tokens if past_length == 0 else tokens[-1:]
                    input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)
                    logits = self.model(
                        input_tensor,
                        kv_cache=kv_cache,
                        position_offset=past_length,
                    )
                else:
                    input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                    logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                if temp <= 0:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()
                else:
                    from inference.sampling import Sampler
                    next_token = Sampler(
                        temperature=temp,
                        top_k=k,
                        top_p=p,
                        repetition_penalty=self.inference_config.repetition_penalty,
                    ).sample(
                        next_token_logits,
                        generated_tokens=tokens,
                        generator=generator,
                    ).item()

                tokens.append(int(next_token))

                if next_token in self.inference_config.stop_token_ids:
                    break

        return tokens

    def generate(
        self,
        prompt: str,
        tokenizer: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        End-to-end string-to-string text generation.
        """
        self._prepare_model()
        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise ValueError("A tokenizer with encode and decode methods is required for text generation")
        prompt_ids = tokenizer.encode(prompt)

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        return tokenizer.decode(output_ids)

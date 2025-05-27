## 4. model.py

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Args:
            d_model: embedding dimension.
            max_len: maximum length of sequences to support.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # register as buffer so it's moved to appropriate device with the module
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape as x with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: model dimension.
            num_heads: number of attention heads.
            d_ff: dimension of feed-forward layer.
            dropout: dropout rate.
        """
        super().__init__()
        # multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        # layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            src_key_padding_mask: (batch_size, seq_len) bool mask, True for padding positions
        Returns:
            output tensor of same shape as x
        """
        # self-attention
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=src_key_padding_mask
        )
        x2 = self.dropout(attn_out)
        x = self.norm1(x + x2)

        # position-wise feed-forward
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x2 = self.dropout(ff)
        x = self.norm2(x + x2)
        return x


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: model dimension.
            num_heads: number of attention heads.
            d_ff: dimension of feed-forward layer.
            dropout: dropout rate.
        """
        super().__init__()
        # masked self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # encoder-decoder attention
        self.enc_dec_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        # layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: target embeddings (batch_size, tgt_len, d_model)
            memory: encoder output (batch_size, src_len, d_model)
            tgt_mask: (tgt_len, tgt_len) mask for causal attention
            tgt_key_padding_mask: (batch_size, tgt_len) bool mask for padding in target
            memory_key_padding_mask: (batch_size, src_len) bool mask for padding in source
        Returns:
            output tensor of shape (batch_size, tgt_len, d_model)
        """
        # masked self-attention
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x2 = self.dropout(attn_out)
        x = self.norm1(x + x2)

        # encoder-decoder attention
        attn_out2, _ = self.enc_dec_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x2 = self.dropout(attn_out2)
        x = self.norm2(x + x2)

        # feed-forward network
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x2 = self.dropout(ff)
        x = self.norm3(x + x2)
        return x


class TransformerModel(nn.Module):
    """Full Transformer model for sequence-to-sequence tasks."""

    def __init__(self, cfg: Config, vocab_size: int) -> None:
        """
        Args:
            cfg: Config object providing hyperparameters.
            vocab_size: size of shared vocabulary for token embeddings.
        """
        super().__init__()
        # read hyperparameters from config
        self.cfg = cfg
        # for MT use "model", for parsing the caller should supply a parsing-specific cfg
        self.num_layers = cfg.get("model.num_layers")
        self.d_model = cfg.get("model.d_model")
        self.d_ff = cfg.get("model.d_ff")
        self.num_heads = cfg.get("model.num_heads")
        self.dropout_rate = cfg.get("model.dropout")
        # sanity check
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )

        # token embedding and positional encoding
        self.token_embed = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout_rate)

        # encoder and decoder stacks
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
                for _ in range(self.num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
                for _ in range(self.num_layers)
            ]
        )

        # output projection (tied with embedding)
        self.output_proj = nn.Linear(self.d_model, vocab_size, bias=False)
        # tie weights
        self.output_proj.weight = self.token_embed.weight

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate a causal mask for masking out future positions."""
        # mask[i,j] = 0 if j<=i else -inf
        mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
        return mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src_ids: (batch_size, src_len) input token IDs
            tgt_ids: (batch_size, tgt_len) target token IDs (for training); if None, returns encoder output
            src_mask: (batch_size, src_len) bool mask, True for real tokens (optional)
            tgt_mask: (batch_size, tgt_len) bool mask, True for real tokens (optional)
        Returns:
            If tgt_ids is provided: logits of shape (batch_size, tgt_len, vocab_size)
            Else: encoder output of shape (batch_size, src_len, d_model)
        """
        # embed and add positional encoding
        x = self.token_embed(src_ids) * math.sqrt(self.d_model)  # (B, S, D)
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)

        # prepare source padding mask for attention (True for pad)
        if src_mask is not None:
            # src_mask: True=real, False=pad -> invert
            src_key_padding_mask = ~src_mask
        else:
            src_key_padding_mask = None

        # encoder forward
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        memory = x  # (B, S, D)

        # if no target, return encoder output
        if tgt_ids is None:
            return memory

        # embed target
        y = self.token_embed(tgt_ids) * math.sqrt(self.d_model)  # (B, T, D)
        y = self.pos_encoder(y)
        y = self.embed_dropout(y)

        # create causal mask for decoder self-attention
        seq_len_t = tgt_ids.size(1)
        device = tgt_ids.device
        causal_mask = self._generate_square_subsequent_mask(seq_len_t).to(device)

        # prepare target padding mask
        if tgt_mask is not None:
            tgt_key_padding_mask = ~tgt_mask
        else:
            tgt_key_padding_mask = None

        # decode
        dec = y
        for layer in self.decoder_layers:
            dec = layer(
                dec,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

        # output projection to vocab logits
        logits = self.output_proj(dec)  # (B, T, V)
        return logits

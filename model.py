## model.py

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_src_mask, make_tgt_mask, positional_encoding, Config


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Args:
            d_model: Total dimension of the model.
            num_heads: Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # scaling factor for dot-product attention
        self.scaling = math.sqrt(self.d_k)

        # Learned projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            query: Tensor of shape (B, L_q, d_model)
            key:   Tensor of shape (B, L_k, d_model)
            value: Tensor of shape (B, L_k, d_model)
            mask:  Optional boolean mask of shape
                   (B, 1, L_q, L_k) or (B, num_heads, L_q, L_k)

        Returns:
            Tensor of shape (B, L_q, d_model)
        """
        B, L_q, _ = query.size()
        _, L_k, _ = key.size()

        # Linear projections
        Q = self.w_q(query)  # (B, L_q, d_model)
        K = self.w_k(key)    # (B, L_k, d_model)
        V = self.w_v(value)  # (B, L_k, d_model)

        # Split heads and transpose
        # New shape: (B, num_heads, L, d_k)
        Q = Q.view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (B, num_heads, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling

        if mask is not None:
            # mask: True for positions we want to attend to
            # we convert to float mask and fill -inf where mask==False
            scores = scores.masked_fill(~mask, float('-1e9'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, L_q, L_k)

        # Attention output
        # (B, num_heads, L_q, d_k)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        # (B, L_q, num_heads, d_k) -> (B, L_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        output = self.w_o(attn_output)  # (B, L_q, d_model)
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: Dimensionality of model input/output.
            d_ff: Dimensionality of inner layer.
            dropout: Dropout rate.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network to each position.

        Args:
            x: Tensor of shape (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model)
        """
        return self.w2(self.dropout(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    """
    Single layer of the Transformer encoder.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: Dimensionality of model.
            num_heads: Number of attention heads.
            d_ff: Dimensionality of feed-forward layer.
            dropout: Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer.

        Args:
            x: Tensor of shape (B, L, d_model)
            src_mask: Encoder padding mask of shape (B, 1, 1, L)

        Returns:
            Tensor of shape (B, L, d_model)
        """
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.drop1(attn_out))

        # Feed-forward sub-layer
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop2(ff_out))
        return x


class DecoderLayer(nn.Module):
    """
    Single layer of the Transformer decoder.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        """
        Args:
            d_model: Dimensionality of model.
            num_heads: Number of attention heads.
            d_ff: Dimensionality of feed-forward layer.
            dropout: Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.

        Args:
            x: Tensor of shape (B, T, d_model)
            enc_out: Encoder output (B, S, d_model)
            src_mask: Encoder padding mask (B, 1, 1, S)
            tgt_mask: Decoder self-attention mask (B, 1, T, T)

        Returns:
            Tensor of shape (B, T, d_model)
        """
        # 1) Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop1(attn1))

        # 2) Encoder-decoder attention
        attn2 = self.enc_dec_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop2(attn2))

        # 3) Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.drop3(ff_out))
        return x


class TransformerModel(nn.Module):
    """
    The full Transformer Model with encoder and decoder stacks.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the Transformer model.

        Args:
            config: Configuration object.
        """
        super(TransformerModel, self).__init__()
        # Model hyperparameters
        model_cfg = config.get("model")
        self.num_layers = int(model_cfg.get("num_layers", 6))
        self.d_model = int(model_cfg.get("d_model", 512))
        self.d_ff = int(model_cfg.get("d_ff", 2048))
        self.num_heads = int(model_cfg.get("num_heads", 8))
        self.dropout_rate = float(model_cfg.get("dropout_rate", 0.1))
        self.share_embeddings = bool(model_cfg.get("share_embeddings", True))
        pe_type = str(model_cfg.get("positional_encoding", "sinusoidal")).lower()

        # Vocabulary size
        self.vocab_size = int(config.get("data.vocab_size", 37000))

        # Embeddings
        self.src_embedding = nn.Embedding(self.vocab_size, self.d_model)
        if self.share_embeddings:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Positional Encoding (sinusoidal or learned)
        max_len = int(model_cfg.get("max_seq_len", 512))
        if pe_type == "sinusoidal":
            # Precompute and register as buffer
            pe = positional_encoding(max_len, self.d_model)  # (1, max_len, d_model)
            self.register_buffer("pos_enc", pe, persistent=False)
            self.learned_pos_enc = None
        elif pe_type == "learned":
            # Learned positional embeddings
            self.learned_pos_enc = nn.Embedding(max_len, self.d_model)
            nn.init.normal_(self.learned_pos_enc.weight, mean=0.0, std=0.02)
            # We won't use buffer pos_enc in this case
            self.register_buffer("pos_enc", torch.zeros(1), persistent=False)
        else:
            raise ValueError(f"Unknown positional_encoding type: {pe_type}")

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.num_layers)
        ])

        # Final linear projection to vocab
        if self.share_embeddings:
            # Tie weights with source embedding
            self.output_linear = nn.Linear(self.d_model, self.vocab_size, bias=False)
            self.output_linear.weight = self.src_embedding.weight
        else:
            self.output_linear = nn.Linear(self.d_model, self.vocab_size)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize parameters with Xavier uniform for linear layers
        and normal distribution for embeddings.
        """
        # Embeddings
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        if not self.share_embeddings:
            nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=self.d_model ** -0.5)

        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Args:
            src: Source token IDs (B, S)
            tgt: Target token IDs (B, T)

        Returns:
            Logits over vocabulary (B, T, vocab_size)
        """
        # Masks
        # Assume pad_id=0; change if your pad token has different ID
        src_mask = make_src_mask(src, pad_id=0)  # (B, 1, 1, S)
        tgt_mask = make_tgt_mask(tgt, pad_id=0)  # (B, 1, T, T)

        # Embedding + positional encoding for src
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)  # (B, S, d_model)
        if self.learned_pos_enc is not None:
            # learned positional
            pos = torch.arange(0, src_emb.size(1), device=src.device).unsqueeze(0)
            src_emb = src_emb + self.learned_pos_enc(pos)
        else:
            # sinusoidal
            src_emb = src_emb + self.pos_enc[:, :src_emb.size(1), :]
        src_out = self.dropout(src_emb)

        # Encoder stack
        for layer in self.encoder_layers:
            src_out = layer(src_out, src_mask)

        # Embedding + positional encoding for tgt
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  # (B, T, d_model)
        if self.learned_pos_enc is not None:
            pos = torch.arange(0, tgt_emb.size(1), device=tgt.device).unsqueeze(0)
            tgt_emb = tgt_emb + self.learned_pos_enc(pos)
        else:
            tgt_emb = tgt_emb + self.pos_enc[:, :tgt_emb.size(1), :]
        dec_out = self.dropout(tgt_emb)

        # Decoder stack
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, src_out, src_mask, tgt_mask)

        # Final projection
        logits = self.output_linear(dec_out)  # (B, T, vocab_size)
        return logits

    def save(self, path: str) -> None:
        """
        Save model state to file.

        Args:
            path: File path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load model state from file.

        Args:
            path: File path from which to load the model.
        """
        state = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state)
        self.eval()

# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Utils


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in "Attention Is All You Need".
    Performs scaled dot-product attention across multiple heads.
    """

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.d_model: int = config.model.d_model
        self.num_heads: int = config.model.num_heads
        self.dropout_rate: float = config.model.dropout_rate

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim: int = self.d_model // self.num_heads
        self.scale: float = 1.0 / math.sqrt(self.head_dim)

        # Linear projections for queries, keys, values
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        # Output projection
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape (batch_size, tgt_len, d_model)
            key:   Tensor of shape (batch_size, src_len, d_model)
            value: Tensor of shape (batch_size, src_len, d_model)
            mask:  Boolean mask Tensor, shape broadcastable to
                   (batch_size, num_heads, tgt_len, src_len)
                   True = allowed, False = masked.
        Returns:
            Tensor of shape (batch_size, tgt_len, d_model)
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # 1) Linear projections
        #    -> (batch_size, seq_len, num_heads, head_dim)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim)

        # 2) Reshape and transpose for attention computation
        #    -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3) Scaled dot-product
        #    -> (batch_size, num_heads, tgt_len, src_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 4) Masking (if provided)
        if mask is not None:
            # mask==True means keep; mask==False means -inf
            scores = scores.masked_fill(~mask, float('-inf'))

        # 5) Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6) Weighted sum of V
        #    -> (batch_size, num_heads, tgt_len, head_dim)
        context = torch.matmul(attn_weights, V)

        # 7) Concatenate heads
        #    -> (batch_size, tgt_len, num_heads * head_dim == d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)

        # 8) Final linear projection
        output = self.W_o(context)
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applies two linear transforms with a ReLU in between, identically at each position.
    """

    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        d_model: int = config.model.d_model
        d_ff: int = config.model.d_ff
        dropout_rate: float = config.model.dropout_rate

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    One layer of the Transformer encoder:
      1) Multi-head self-attention
      2) Add & Norm
      3) Position-wise feed-forward
      4) Add & Norm
    """

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        d_model: int = config.model.d_model
        dropout_rate: float = config.model.dropout_rate

        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward sub-layer
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    """
    One layer of the Transformer decoder:
      1) Masked multi-head self-attention
      2) Add & Norm
      3) Multi-head encoder-decoder attention
      4) Add & Norm
      5) Position-wise feed-forward
      6) Add & Norm
    """

    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        d_model: int = config.model.d_model
        dropout_rate: float = config.model.dropout_rate

        self.self_attn = MultiHeadAttention(config)
        self.enc_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        # 1) Masked self-attention
        attn1 = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # 2) Encoder-decoder attention
        attn2 = self.enc_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # 3) Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TransformerModel(nn.Module):
    """
    The full Transformer model, including:
      - Token embeddings (+ optional weight sharing with output layer)
      - Positional encodings (sinusoidal or learned)
      - Stacks of encoder and decoder layers
      - Final linear projection to vocabulary logits
    """

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config

        # Hyperparameters
        d_model: int = config.model.d_model
        vocab_size: int = config.data.spm_vocab_size
        dropout_rate: float = config.model.dropout_rate
        share_embeddings: bool = config.model.share_embeddings
        pos_enc_type: str = config.model.positional_encoding

        # Token embedding & scaling
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

        # Positional encodings
        self.max_seq_len = getattr(config.model, "max_seq_len", 5000)
        if pos_enc_type == "sinusoidal":
            pe = torch.zeros(self.max_seq_len, d_model)
            position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
            self.register_buffer("positional_encoding", pe)
            self.positional_embedding = None
        else:
            # Learned positional embeddings
            self.positional_encoding = None
            self.positional_embedding = nn.Embedding(self.max_seq_len, d_model)

        # Dropout on summed embeddings
        self.dropout = nn.Dropout(dropout_rate)

        # Encoder & decoder stacks
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.model.encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.model.decoder_layers)]
        )

        # Final output projection
        if share_embeddings:
            self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
            self.output_proj.weight = self.embedding.weight
        else:
            self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for a batch of source and target sequences.

        Args:
            src: (batch_size, src_len) source token IDs
            tgt: (batch_size, tgt_len) target token IDs (teacher forcing)

        Returns:
            logits: (batch_size, tgt_len, vocab_size)
        """
        # 1) Build masks
        src_mask, tgt_mask = Utils.make_masks(src, tgt, self.config)

        # 2) Embed & add positional encoding (source)
        src_emb = self.embedding(src) * self.scale
        src_len = src.size(1)
        if self.positional_encoding is not None:
            pe_src = self.positional_encoding[:, :src_len, :].to(src_emb.device)
            src_emb = src_emb + pe_src
        else:
            pos_ids = torch.arange(src_len, device=src.device).unsqueeze(0)
            src_emb = src_emb + self.positional_embedding(pos_ids)
        src_emb = self.dropout(src_emb)

        # 3) Encoder stack
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        # 4) Embed & add positional encoding (target)
        tgt_emb = self.embedding(tgt) * self.scale
        tgt_len = tgt.size(1)
        if self.positional_encoding is not None:
            pe_tgt = self.positional_encoding[:, :tgt_len, :].to(tgt_emb.device)
            tgt_emb = tgt_emb + pe_tgt
        else:
            pos_ids = torch.arange(tgt_len, device=tgt.device).unsqueeze(0)
            tgt_emb = tgt_emb + self.positional_embedding(pos_ids)
        tgt_emb = self.dropout(tgt_emb)

        # 5) Decoder stack
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        # 6) Final linear projection
        logits = self.output_proj(dec_out)
        return logits

## model.py

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in
    'Attention Is All You Need', Section 3.5.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum input sequence length.
            dropout (float): Dropout rate to apply after adding positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it's saved with the model, but not a parameter
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Positionally encoded embeddings of same shape.
        """
        seq_len = x.size(1)
        # Add positional encoding up to seq_len
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as in 'Attention Is All You Need', Section 3.2.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): Total dimension of the model.
            num_heads (int): Number of parallel attention heads.
            dropout (float): Dropout probability on attention weights.
        """
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for query, key, value, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query (Tensor): shape (batch_size, len_q, d_model)
            key (Tensor):   shape (batch_size, len_k, d_model)
            value (Tensor): shape (batch_size, len_v, d_model)
            mask (Tensor, optional): 
                Boolean mask tensor. Shape broadcastable to
                (batch_size, num_heads, len_q, len_k). True for positions to keep.

        Returns:
            Tensor: Attention output of shape (batch_size, len_q, d_model).
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        def _shape(x: torch.Tensor, linear: nn.Linear):
            x = linear(x)  # (B, L, d_model)
            # (B, L, num_heads, d_k)
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            # (B, num_heads, L, d_k)
            return x.transpose(1, 2)

        q = _shape(query, self.w_q)
        k = _shape(key, self.w_k)
        v = _shape(value, self.w_v)

        # Scaled dot-product attention
        # scores: (B, num_heads, len_q, len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask == True are valid; mask == False are masked
            scores = scores.masked_fill(~mask, float('-1e9'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # context: (B, num_heads, len_q, d_k)
        context = torch.matmul(attn, v)
        # concat heads: (B, len_q, num_heads * d_k = d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # final linear
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network, two linear layers with ReLU in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): Input and output dimension.
            d_ff (int): Inner-layer dimension.
            dropout (float): Dropout probability after first linear.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: shape (batch_size, seq_len, d_model)
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    One layer of the Transformer encoder, consisting of:
      (1) Multi-head self-attention
      (2) Position-wise feed-forward
    Each sub-layer has a residual connection followed by layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Feed-forward inner dimension.
            dropout (float): Dropout rate for both sub-layers.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (batch_size, seq_len, d_model)
            src_mask (Tensor, optional): source padding mask,
                broadcastable to (batch_size, num_heads, seq_len, seq_len)

        Returns:
            Tensor: shape (batch_size, seq_len, d_model)
        """
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward sub-layer
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    """
    One layer of the Transformer decoder, consisting of:
      (1) Masked multi-head self-attention
      (2) Multi-head encoder-decoder attention
      (3) Position-wise feed-forward
    Each sub-layer has a residual connection followed by layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Feed-forward inner dimension.
            dropout (float): Dropout rate for all sub-layers.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Decoder input embeddings, shape (batch_size, tgt_len, d_model)
            memory (Tensor): Encoder output, shape (batch_size, src_len, d_model)
            tgt_mask (Tensor, optional): Mask for decoder self-attention,
                broadcastable to (batch_size, num_heads, tgt_len, tgt_len)
            memory_mask (Tensor, optional): Mask for encoder-decoder attention,
                broadcastable to (batch_size, num_heads, tgt_len, src_len)

        Returns:
            Tensor: shape (batch_size, tgt_len, d_model)
        """
        # Masked self-attention
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        # Encoder-decoder attention
        enc_dec_out = self.enc_dec_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout(enc_dec_out))
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TransformerModel(nn.Module):
    """
    The full Transformer model for sequence-to-sequence tasks,
    combining encoder and decoder stacks.
    """

    def __init__(self, config: dict, vocab_size: int):
        """
        Args:
            config (dict): Model hyperparameters, expect keys:
                - d_model: int
                - d_ff: int
                - num_layers: int
                - num_heads: int
                - dropout: float
                - max_position_embeddings: int
            vocab_size (int): Size of the shared vocabulary (source+target).
        """
        super(TransformerModel, self).__init__()
        # Model hyperparameters
        self.d_model = int(config.get("d_model", 512))
        self.d_ff = int(config.get("d_ff", 2048))
        self.num_layers = int(config.get("num_layers", 6))
        self.num_heads = int(config.get("num_heads", 8))
        self.dropout = float(config.get("dropout", 0.1))
        self.max_pos_emb = int(config.get("max_position_embeddings", 512))

        # Token embeddings and weight tying
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.d_model,
            padding_idx=0
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.max_pos_emb,
            dropout=self.dropout
        )

        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Final normalization on decoder output
        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def _generate_padding_mask(
        self, seq: torch.Tensor, pad_idx: int = 0
    ) -> torch.Tensor:
        """
        Creates padding mask for sequences.

        Args:
            seq (Tensor): shape (batch_size, seq_len)
            pad_idx (int): padding token index

        Returns:
            Tensor: mask of shape (batch_size, 1, 1, seq_len),
                    True for non-pad tokens.
        """
        # True at positions that are not padding
        mask = seq.ne(pad_idx)
        # (batch_size, 1, 1, seq_len)
        return mask.unsqueeze(1).unsqueeze(2)

    def _generate_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Creates a causal (lower-triangular) mask for decoder self-attention.

        Args:
            size (int): target sequence length
            device (torch.device): device for resulting mask

        Returns:
            Tensor: mask of shape (1, 1, size, size), True in lower triangle.
        """
        # Lower triangular matrix
        subsequent = torch.tril(torch.ones((size, size), device=device)).bool()
        # (1,1,size,size) for broadcasting
        return subsequent.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Args:
            src_ids (Tensor): source token IDs, shape (batch_size, src_len)
            tgt_ids (Tensor): target token IDs, shape (batch_size, tgt_len)

        Returns:
            Tensor: logits over vocabulary, shape (batch_size, tgt_len, vocab_size)
        """
        device = src_ids.device

        # Masks
        src_mask = self._generate_padding_mask(src_ids, pad_idx=0)
        tgt_pad_mask = self._generate_padding_mask(tgt_ids, pad_idx=0)
        seq_len = tgt_ids.size(1)
        subsequent_mask = self._generate_subsequent_mask(seq_len, device)
        # Combine pad and causal masks for decoder self-attention
        tgt_mask = tgt_pad_mask & subsequent_mask  # (B,1,tgt_len,tgt_len)
        memory_mask = src_mask  # same attention mask for encoder-decoder

        # Embedding + positional encoding
        # Scale embeddings by sqrt(d_model)
        src_emb = self.embedding(src_ids) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Encoder
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # Decoder
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, memory_mask)

        # Final layer norm
        dec_output = self.final_norm(dec_output)

        # Project to vocabulary (tied weights with embedding)
        logits = F.linear(dec_output, self.embedding.weight)  # (B, tgt_len, vocab_size)
        return logits

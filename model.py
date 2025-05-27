# model.py

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for token embeddings.

    Attributes:
        dropout: Dropout layer applied after adding positional encodings.
        pe:      Positional encoding buffer of shape (max_len, d_model).
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model:      Dimension of the embeddings.
            dropout_rate: Dropout probability.
            max_len:      Maximum sequence length to support.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Create positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it's saved but not trained
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor of same shape as x with positional encodings added and dropout applied.
        """
        seq_len = x.size(1)
        # Add positional encoding and apply dropout
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Attributes:
        scale: Scaling factor 1/sqrt(d_k).
    """

    def __init__(self, d_k: int) -> None:
        """
        Args:
            d_k: Dimension of the key (and query) vectors.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Q:    Query tensor of shape (batch, n_heads, seq_len_q, d_k).
            K:    Key tensor   of shape (batch, n_heads, seq_len_k, d_k).
            V:    Value tensor of shape (batch, n_heads, seq_len_v, d_v).
            mask: Optional mask tensor broadcastable to (batch, n_heads, seq_len_q, seq_len_k).

        Returns:
            Attention output of shape (batch, n_heads, seq_len_q, d_v).
        """
        # (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask == 0 means padded or future position
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    Attributes:
        n_heads:       Number of parallel attention heads.
        d_k:           Dimension of keys/queries per head.
        d_v:           Dimension of values per head.
        W_Q, W_K, W_V: Linear projections for queries, keys, values.
        W_O:           Output linear projection.
        attention:     Scaled dot-product attention module.
        dropout:       Dropout layer on attention output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout_rate: float
    ) -> None:
        """
        Args:
            d_model:      Total dimension of the model.
            n_heads:      Number of attention heads.
            d_k:          Dimension of keys/queries per head.
            d_v:          Dimension of values per head.
            dropout_rate: Dropout probability on attention output.
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        # Combined projections to speed up computation
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(d_k)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Q:    Queries of shape (batch, seq_len_q, d_model).
            K:    Keys    of shape (batch, seq_len_k, d_model).
            V:    Values  of shape (batch, seq_len_v, d_model).
            mask: Optional mask of shape (batch, 1, seq_len_q, seq_len_k).

        Returns:
            Tensor of shape (batch, seq_len_q, d_model).
        """
        batch_size = Q.size(0)

        # Linear projections and reshape for multi-head: (batch, n_heads, seq_len, d_k)
        def project(x, proj):
            x = proj(x)  # (batch, seq_len, n_heads*d_k or d_v)
            x = x.view(batch_size, -1, self.n_heads, x.size(-1) // self.n_heads)
            # transpose to (batch, n_heads, seq_len, d_k/d_v)
            return x.transpose(1, 2)

        q = project(Q, self.W_Q)
        k = project(K, self.W_K)
        v = project(V, self.W_V)

        # Apply attention on all the projected vectors in batch
        # mask shape must be broadcastable to (batch, n_heads, seq_len_q, seq_len_k)
        if mask is not None:
            # mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len_q, seq_len_k)
            mask = mask.expand(batch_size, self.n_heads, mask.size(-2), mask.size(-1))

        head_out = self.attention(q, k, v, mask=mask)
        # Concatenate heads and put through final linear projection
        # head_out: (batch, n_heads, seq_len_q, d_v) -> (batch, seq_len_q, n_heads*d_v)
        head_out = head_out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.W_O(head_out)
        return self.dropout(output)


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.

    Attributes:
        fc1, fc2: Two linear layers.
        dropout:  Dropout layer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
        """
        Args:
            d_model:      Input and output dimension.
            d_ff:         Inner-layer dimension.
            dropout_rate: Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of same shape after feed-forward transformation.
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Single layer of the Transformer encoder, consisting of:
      (1) Multi-head self-attention + residual + layer norm
      (2) Position-wise feed-forward + residual + layer norm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout_rate: float
    ) -> None:
        """
        Args:
            d_model:      Model dimensionality.
            n_heads:      Number of attention heads.
            d_k:          Key/query dimension per head.
            d_v:          Value dimension per head.
            d_ff:         Feed-forward inner dimension.
            dropout_rate: Dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        Input tensor of shape (batch, seq_len, d_model).
            src_mask: Source padding mask of shape (batch, 1, 1, seq_len).

        Returns:
            Tensor of same shape after applying encoder layer.
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
    Single layer of the Transformer decoder, consisting of:
      (1) Masked multi-head self-attention + residual + layer norm
      (2) Encoder-decoder attention       + residual + layer norm
      (3) Position-wise feed-forward      + residual + layer norm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout_rate: float
    ) -> None:
        """
        Args:
            d_model:      Model dimensionality.
            n_heads:      Number of attention heads.
            d_k:          Key/query dimension per head.
            d_v:          Value dimension per head.
            d_ff:         Feed-forward inner dimension.
            dropout_rate: Dropout probability.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, dropout_rate)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:        Decoder input of shape (batch, tgt_len, d_model).
            memory:   Encoder output of shape (batch, src_len, d_model).
            src_mask: Source mask of shape (batch, 1, 1, src_len).
            tgt_mask: Target mask of shape (batch, 1, tgt_len, tgt_len).

        Returns:
            Tensor of shape (batch, tgt_len, d_model).
        """
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Encoder-decoder attention
        attn2 = self.enc_dec_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TransformerModel(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    Public methods:
      - forward(src, tgt, src_mask, tgt_mask) -> logits
      - generate(src, max_len, beam_size) -> List[List[int]]
    """

    def __init__(self, model_params: Dict) -> None:
        """
        Initializes the TransformerModel.

        Args:
            model_params: Dictionary containing:
              - encoder_layers: int
              - decoder_layers: int
              - d_model: int
              - d_ff: int
              - n_heads: int
              - d_k: int
              - d_v: int
              - dropout_rate: float
              - vocab_size: int
              - pad_idx: int (optional, default=0)
              - bos_idx: int (optional, default=1)
              - eos_idx: int (optional, default=2)
              - length_penalty: float (optional, default=0.6)
        """
        super(TransformerModel, self).__init__()
        # Extract hyperparameters
        self.encoder_layers = int(model_params["encoder_layers"])
        self.decoder_layers = int(model_params["decoder_layers"])
        self.d_model = int(model_params["d_model"])
        self.d_ff = int(model_params["d_ff"])
        self.n_heads = int(model_params["n_heads"])
        self.d_k = int(model_params["d_k"])
        self.d_v = int(model_params["d_v"])
        dropout_rate = float(model_params["dropout_rate"])

        # Special tokens and length penalty
        self.vocab_size = int(model_params["vocab_size"])
        self.pad_idx = int(model_params.get("pad_idx", 0))
        self.bos_idx = int(model_params.get("bos_idx", 1))
        self.eos_idx = int(model_params.get("eos_idx", 2))
        self.length_penalty = float(model_params.get("length_penalty", 0.6))

        # Token embedding (shared by encoder and decoder)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_rate)

        # Build encoder and decoder stacks
        self.encoder_stack = nn.ModuleList([
            EncoderLayer(
                self.d_model,
                self.n_heads,
                self.d_k,
                self.d_v,
                self.d_ff,
                dropout_rate
            )
            for _ in range(self.encoder_layers)
        ])
        self.decoder_stack = nn.ModuleList([
            DecoderLayer(
                self.d_model,
                self.n_heads,
                self.d_k,
                self.d_v,
                self.d_ff,
                dropout_rate
            )
            for _ in range(self.decoder_layers)
        ])

        # Final linear layer (pre-softmax) tied to embedding weights
        self.generator = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.generator.weight = self.embed.weight  # weight tying

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs the forward pass of the Transformer.

        Args:
            src:      Source token IDs, shape (batch, src_len).
            tgt:      Target token IDs (input), shape (batch, tgt_len).
            src_mask: Source padding mask, shape (batch, 1, 1, src_len).
            tgt_mask: Target mask, shape (batch, 1, tgt_len, tgt_len).

        Returns:
            Logits over vocabulary, shape (batch, tgt_len, vocab_size).
        """
        # Embedding + positional encoding for source
        src_emb = self.embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Pass through encoder stack
        memory = src_emb
        for layer in self.encoder_stack:
            memory = layer(memory, src_mask)

        # Embedding + positional encoding for target
        tgt_emb = self.embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Pass through decoder stack
        output = tgt_emb
        for layer in self.decoder_stack:
            output = layer(output, memory, src_mask, tgt_mask)

        # Final linear projection to vocabulary
        logits = self.generator(output)
        return logits

    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        beam_size: int
    ) -> List[List[int]]:
        """
        Generates sequences from the source inputs using beam search.

        Args:
            src:      Source token IDs, shape (batch, src_len).
            max_len:  Maximum length of generated sequences.
            beam_size: Beam size for beam search.

        Returns:
            List of generated token ID sequences (with BOS and EOS).
        """
        device = src.device
        batch_size, src_len = src.size()

        # Create source mask
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)

        # Encode source once
        with torch.no_grad():
            src_emb = self.embed(src) * math.sqrt(self.d_model)
            src_emb = self.pos_encoder(src_emb)
            memory = src_emb
            for layer in self.encoder_stack:
                memory = layer(memory, src_mask)

        all_hypotheses: List[List[int]] = []

        # Beam search per example
        for i in range(batch_size):
            mem_i = memory[i : i + 1]      # (1, src_len, d_model)
            src_mask_i = src_mask[i : i + 1]  # (1,1,1,src_len)

            # Initialize beams: list of (seq, score)
            beams = [([self.bos_idx], 0.0)]
            completed: List[Tuple[List[int], float]] = []

            for _ in range(max_len):
                candidates: List[Tuple[List[int], float]] = []
                for seq, score in beams:
                    if seq[-1] == self.eos_idx:
                        # Already ended
                        candidates.append((seq, score))
                        continue

                    # Prepare decoder input
                    tgt_seq = torch.tensor([seq], dtype=torch.long, device=device)
                    # Build target mask (padding + subsequent masking)
                    seq_len = tgt_seq.size(1)
                    # Padding mask
                    tgt_pad = (tgt_seq != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (1,1,1,seq_len)
                    # Look-ahead mask
                    look_ahead = torch.tril(
                        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
                    ).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,seq_len)
                    tgt_mask = tgt_pad & look_ahead  # (1,1,seq_len,seq_len)

                    # Forward step
                    logits = self.forward(mem_i.new_full((1, src_len), 0), tgt_seq, src_mask_i, tgt_mask)
                    # Note: src passed as dummy since forward uses only memory from encoder above
                    # We bypass the initial encoding by feeding memory directly in DecoderLayer loop.
                    # Simplify: re-run decoder only
                    # A more efficient approach would factor out encoder/decoder in separate methods.

                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # (vocab_size,)

                    # Get top-k candidates
                    top_probs, top_ids = torch.topk(log_probs, beam_size)
                    for k in range(beam_size):
                        next_token = top_ids[k].item()
                        next_score = score + top_probs[k].item()
                        candidates.append((seq + [next_token], next_score))

                # Select top beam_size beams with length penalty
                def norm_score(sc, s):
                    return sc / (len(s) ** self.length_penalty)

                ordered = sorted(
                    candidates,
                    key=lambda x: norm_score(x[1], x[0]),
                    reverse=True
                )
                beams = ordered[:beam_size]

                # If all beams are completed, stop early
                if all(seq[-1] == self.eos_idx for seq, _ in beams):
                    break

            # Add completed beams (or current beams if none finished)
            best_seq, best_score = max(
                beams if beams else completed,
                key=lambda x: x[1] / (len(x[0]) ** self.length_penalty)
            )
            all_hypotheses.append(best_seq)

        return all_hypotheses

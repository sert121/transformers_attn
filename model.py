## model.py

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
import utils


class PositionalEncoding(nn.Module):
    """Injects sinusoidal positional encodings."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """
        Args:
            d_model: Embedding dimension.
            dropout_rate: Dropout probability.
            max_len: Maximum length of sequences to expect.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Create positional encoding matrix of size (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape with positional encodings added.
        """
        seq_len = x.size(1)
        # pe[:, :seq_len, :] has shape (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Implements scaled dot-product attention."""

    def __init__(self, dropout_rate: float) -> None:
        """
        Args:
            dropout_rate: Dropout probability on attention weights.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            Q: Queries of shape (batch, num_heads, seq_len_q, d_k)
            K: Keys    of shape (batch, num_heads, seq_len_k, d_k)
            V: Values  of shape (batch, num_heads, seq_len_k, d_v)
            mask: Mask tensor broadcastable to (batch, num_heads, seq_len_q, seq_len_k)
                  with True in positions to mask.
        Returns:
            Attention output of shape (batch, num_heads, seq_len_q, d_v)
        """
        d_k = Q.size(-1)
        # scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask==True indicates positions to set to -inf
            scores = scores.masked_fill(mask, float("-1e9"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # output: (batch, num_heads, seq_len_q, d_v)
        output = torch.matmul(attn, V)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float) -> None:
        """
        Args:
            d_model: Total dimension of model.
            num_heads: Number of attention heads.
            dropout_rate: Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        # Learned linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape (batch, seq_len_q, d_model)
            key:   Tensor of shape (batch, seq_len_k, d_model)
            value: Tensor of shape (batch, seq_len_k, d_model)
            mask:  Optional mask for attention weights.
        Returns:
            Tensor of shape (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_Q(query)  # (batch, seq_len_q, d_model)
        K = self.W_K(key)    # (batch, seq_len_k, d_model)
        V = self.W_V(value)  # (batch, seq_len_k, d_model)

        # Split into heads and transpose for attention: (batch, num_heads, seq_len, d_k)
        def shape(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = shape(Q)
        K = shape(K)
        V = shape(V)

        if mask is not None:
            # mask needs to broadcast over heads
            mask = mask.to(Q.device)

        # Apply attention on all heads
        attn_out = self.attention(Q, K, V, mask)  # (batch, num_heads, seq_len_q, d_v)

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous()  # (batch, seq_len_q, num_heads, d_v)
        attn_out = attn_out.view(batch_size, -1, self.num_heads * self.d_v)  # (batch, seq_len_q, d_model)

        # Final linear projection
        output = self.W_O(attn_out)  # (batch, seq_len_q, d_model)
        return self.dropout(output)


class PositionwiseFeedForward(nn.Module):
    """Implements the position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
        """
        Args:
            d_model: Input and output dimension.
            d_ff: Inner-layer dimension.
            dropout_rate: Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of same shape.
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Single layer of the Transformer encoder."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float) -> None:
        """
        Args:
            d_model: Hidden dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward inner dimension.
            dropout_rate: Dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        Input tensor (batch, seq_len, d_model)
            src_mask: Padding mask (batch, 1, 1, seq_len)
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward sub-layer
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class DecoderLayer(nn.Module):
    """Single layer of the Transformer decoder."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float) -> None:
        """
        Args:
            d_model: Hidden dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward inner dimension.
            dropout_rate: Dropout probability.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:        Input tensor (batch, tgt_len, d_model)
            memory:   Encoder outputs (batch, src_len, d_model)
            src_mask: Encoder padding mask (batch, 1, 1, src_len)
            tgt_mask: Combined mask for target (batch, 1, tgt_len, tgt_len)
        Returns:
            Tensor of shape (batch, tgt_len, d_model)
        """
        # 1) Masked self-attention
        attn1 = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn1))

        # 2) Encoder-decoder attention
        attn2 = self.enc_dec_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout2(attn2))

        # 3) Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x


class Encoder(nn.Module):
    """Transformer encoder composed of a stack of N layers."""

    def __init__(self, config: Config) -> None:
        """
        Args:
            config: Configuration object.
        """
        super(Encoder, self).__init__()
        self.config = config

        self.N = config.get("model.encoder_layers", 6)
        self.d_model = config.get("model.d_model", 512)
        self.d_ff = config.get("model.d_ff", 2048)
        self.num_heads = config.get("model.num_heads", 8)
        self.dropout_rate = config.get("model.dropout_rate", 0.1)
        self.vocab_size = config.get("data.vocab.size", 37000)

        # Token embedding + positional encoding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate)

        # Stacked encoder layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
                for _ in range(self.N)
            ]
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src:      Source token IDs (batch, src_len)
            src_mask: Padding mask for source (batch, 1, 1, src_len)
        Returns:
            Encoder output of shape (batch, src_len, d_model)
        """
        # Embedding + scale + positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """Transformer decoder composed of a stack of N layers."""

    def __init__(self, config: Config) -> None:
        """
        Args:
            config: Configuration object.
        """
        super(Decoder, self).__init__()
        self.config = config

        self.N = config.get("model.decoder_layers", 6)
        self.d_model = config.get("model.d_model", 512)
        self.d_ff = config.get("model.d_ff", 2048)
        self.num_heads = config.get("model.num_heads", 8)
        self.dropout_rate = config.get("model.dropout_rate", 0.1)
        self.vocab_size = config.get("data.vocab.size", 37000)

        # Token embedding + positional encoding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate)

        # Stacked decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
                for _ in range(self.N)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tgt:      Target token IDs (batch, tgt_len)
            memory:   Encoder outputs     (batch, src_len, d_model)
            src_mask: Encoder padding mask (batch, 1, 1, src_len)
            tgt_mask: Combined target mask (batch, 1, tgt_len, tgt_len)
        Returns:
            Decoder output of shape (batch, tgt_len, d_model)
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class TransformerModel(nn.Module):
    """Full Transformer model including encoder, decoder, and generation."""

    def __init__(self, config: Config) -> None:
        """
        Args:
            config: Configuration object.
        """
        super(TransformerModel, self).__init__()
        self.config = config
        self.device = utils.get_device(config)

        # Instantiate encoder and decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Final linear projection to vocabulary
        d_model = config.get("model.d_model", 512)
        vocab_size = config.get("data.vocab.size", 37000)
        share_embed = config.get("model.share_embedding_and_softmax", True)

        # Weight tying if requested
        if share_embed:
            # Tie embeddings between encoder and decoder
            self.decoder.embedding.weight = self.encoder.embedding.weight

        self.final_proj = nn.Linear(d_model, vocab_size)
        if share_embed:
            # Tie final projection weight to embedding weight
            self.final_proj.weight = self.encoder.embedding.weight

        # Special token IDs (assumed as in tokenizer)
        self.pad_id = config.get("data.pad_id", 0)
        self.sos_id = config.get("data.sos_id", 1)
        self.eos_id = config.get("data.eos_id", 2)
        self.alpha = config.get("inference.length_penalty_alpha", 0.0)

        # Initialize all weights
        self.apply(utils.initialize_weights)
        self.to(self.device)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src:      Source tokens (batch, src_len)
            src_mask: Source padding mask
        Returns:
            Encoder outputs (batch, src_len, d_model)
        """
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tgt:      Target tokens (batch, tgt_len)
            memory:   Encoder outputs  (batch, src_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target combined mask
        Returns:
            Decoder outputs (batch, tgt_len, d_model)
        """
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard forward pass for training.

        Args:
            src:      Source tokens
            tgt:      Target tokens (including <s> at start)
            src_mask: Source padding mask
            tgt_mask: Target combined mask
        Returns:
            Logits over vocabulary (batch, tgt_len, vocab_size)
        """
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)
        logits = self.final_proj(dec_out)
        return logits

    def generate(self, src: torch.Tensor, beam_size: int, max_len: int) -> List[List[int]]:
        """
        Beam search generation for inference.

        Args:
            src:       Source token IDs (batch, src_len)
            beam_size: Beam width
            max_len:   Maximum target length (includes <s>)
        Returns:
            List of generated token ID lists (excluding <s> and </s>).
        """
        batch_size, src_len = src.size()
        device = src.device

        # Compute source mask and encoder outputs
        src_mask = utils.create_padding_mask(src, self.pad_id).to(device)
        memory = self.encode(src, src_mask)  # (batch, src_len, d_model)

        all_results: List[List[int]] = []

        # Process each example independently
        for i in range(batch_size):
            memory_i = memory[i : i + 1]           # (1, src_len, d_model)
            src_mask_i = src_mask[i : i + 1]       # (1,1,1,src_len)

            # Initialize beams: (sequence, cum_log_prob, completed_flag)
            beams = [([self.sos_id], 0.0, False)]

            for _ in range(max_len):
                candidates = []
                # Expand each beam
                for seq, cum_score, completed in beams:
                    if completed:
                        candidates.append((seq, cum_score, True))
                        continue

                    # Prepare decoder input and mask
                    tgt_input = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                    tgt_mask = utils.create_combined_mask(tgt_input, self.pad_id).to(device)

                    # Decode one step
                    dec_out = self.decode(tgt_input, memory_i, src_mask_i, tgt_mask)
                    logits = self.final_proj(dec_out[:, -1, :])  # (1, vocab_size)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)

                    # Select top-k token predictions
                    topk_logps, topk_ids = torch.topk(log_probs, beam_size)
                    for k in range(beam_size):
                        token = int(topk_ids[k].item())
                        new_seq = seq + [token]
                        new_score = cum_score + float(topk_logps[k].item())
                        is_done = token == self.eos_id
                        candidates.append((new_seq, new_score, is_done))

                # Apply length normalization and pick top beams
                def norm_score(item):
                    seq, score, _ = item
                    length = len(seq)
                    if self.alpha > 0.0:
                        penalty = length ** self.alpha
                    else:
                        penalty = float(length)
                    return score / penalty

                # Sort by normalized score descending
                candidates.sort(key=norm_score, reverse=True)
                beams = candidates[:beam_size]

                # If all beams are completed, stop early
                if all(c for _, _, c in beams):
                    break

            # Choose best beam, strip <s> and tokens after </s>
            best_seq = beams[0][0]
            # Remove initial <s>
            if best_seq and best_seq[0] == self.sos_id:
                best_seq = best_seq[1:]
            # Truncate at </s>
            if self.eos_id in best_seq:
                idx = best_seq.index(self.eos_id)
                best_seq = best_seq[:idx]

            all_results.append(best_seq)

        return all_results

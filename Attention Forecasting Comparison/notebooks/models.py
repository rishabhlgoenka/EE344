"""
models.py — Shared model definitions for EE344 Honours Project

Three attention-based forecasting architectures with matched parameter counts (~95K).
  M1: Seq2SeqBahdanau      — LSTM encoder-decoder, Bahdanau (additive) attention
  M2: SelfAttentionLSTM    — LSTM encoder, self-attention pre-processing, autoregressive decoder
  M3: TransformerForecaster — Full Transformer encoder-decoder with cross-attention
"""

import math
import torch
import torch.nn as nn

# ════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════

class BahdanauAttention(nn.Module):
    """score = v^T tanh(W_enc·h_enc + W_dec·h_dec)"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        score = self.v(torch.tanh(
            self.W_enc(enc_outputs) + self.W_dec(dec_hidden).unsqueeze(1)
        ))
        weights = torch.softmax(score.squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learnable parameters)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ════════════════════════════════════════════════════════════════
# M1: LSTM Encoder-Decoder with Bahdanau (Additive) Attention
# ════════════════════════════════════════════════════════════════

class Seq2SeqBahdanau(nn.Module):
    """LSTM encoder-decoder with Bahdanau attention.

    Autoregressive decoder: each step receives [context_vector, prev_prediction].
    Training uses teacher forcing (controlled by tf_ratio).
    Inference uses own predictions as feedback.
    """

    def __init__(self, n_features, hidden_dim, enc_layers, horizon,
                 target_idx, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.target_idx = target_idx
        self.encoder = nn.LSTM(
            n_features, hidden_dim, enc_layers,
            batch_first=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.attention = BahdanauAttention(hidden_dim)
        self.decoder_cell = nn.LSTMCell(hidden_dim + 1, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, targets=None, tf_ratio=0.0, return_attn=False):
        """
        x:        (B, lookback, n_features)
        targets:  (B, horizon) — ground truth; required when tf_ratio > 0
        tf_ratio: probability of using ground truth at each decode step
        """
        enc_out, (h_n, c_n) = self.encoder(x)
        dec_h, dec_c = h_n[-1], c_n[-1]
        prev_y = x[:, -1, self.target_idx].unsqueeze(-1)

        preds = []
        attn_list = [] if return_attn else None
        for t in range(self.horizon):
            ctx, w = self.attention(dec_h, enc_out)
            dec_input = torch.cat([ctx, prev_y], dim=-1)
            dec_h, dec_c = self.decoder_cell(dec_input, (dec_h, dec_c))
            out = self.fc_out(self.dropout(dec_h))
            preds.append(out)
            if return_attn:
                attn_list.append(w)

            use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
            prev_y = targets[:, t].unsqueeze(-1) if use_teacher else out.detach()

        preds = torch.cat(preds, dim=-1)
        if return_attn:
            return preds, torch.stack(attn_list, dim=1)
        return preds


# ════════════════════════════════════════════════════════════════
# M2: LSTM Encoder + Self-Attention + Autoregressive Decoder
# ════════════════════════════════════════════════════════════════

class SelfAttentionLSTM(nn.Module):
    """LSTM encoder → self-attention refinement → autoregressive LSTM decoder.

    Self-attention (scaled dot-product, single head) is applied over encoder
    hidden states *before* decoding.  The decoder uses Bahdanau attention over
    the self-attended states at each step — identical mechanism to M1, so the
    only controlled variable is the self-attention pre-processing layer.

    Requires teacher forcing + scheduled sampling (same interface as M1).
    ~94,921 params with hidden_dim=60.
    """

    def __init__(self, n_features, hidden_dim, enc_layers, horizon,
                 target_idx, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.target_idx = target_idx
        self.attn_scale = math.sqrt(hidden_dim)

        self.encoder = nn.LSTM(
            n_features, hidden_dim, enc_layers,
            batch_first=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attention = BahdanauAttention(hidden_dim)
        self.decoder_cell = nn.LSTMCell(hidden_dim + 1, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, targets=None, tf_ratio=0.0, return_attn=False):
        """Same interface as Seq2SeqBahdanau for drop-in use."""
        enc_out, (h_n, c_n) = self.encoder(x)

        Q = self.W_q(enc_out)
        K = self.W_k(enc_out)
        V = self.W_v(enc_out)
        sa_scores = torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale
        sa_weights = torch.softmax(sa_scores, dim=-1)
        attended = torch.bmm(sa_weights, V)

        dec_h, dec_c = h_n[-1], c_n[-1]
        prev_y = x[:, -1, self.target_idx].unsqueeze(-1)

        preds = []
        attn_list = [] if return_attn else None
        for t in range(self.horizon):
            ctx, w = self.attention(dec_h, attended)
            dec_input = torch.cat([ctx, prev_y], dim=-1)
            dec_h, dec_c = self.decoder_cell(dec_input, (dec_h, dec_c))
            out = self.fc_out(self.dropout(dec_h))
            preds.append(out)
            if return_attn:
                attn_list.append(w)

            use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
            prev_y = targets[:, t].unsqueeze(-1) if use_teacher else out.detach()

        preds = torch.cat(preds, dim=-1)
        if return_attn:
            return preds, torch.stack(attn_list, dim=1), sa_weights
        return preds


# ════════════════════════════════════════════════════════════════
# M3: Full Transformer Encoder-Decoder with Cross-Attention
# ════════════════════════════════════════════════════════════════

class TransformerForecaster(nn.Module):
    """Transformer encoder-decoder with learned query tokens.

    Encoder: input projection + sinusoidal PE + TransformerEncoder.
    Decoder: 96 learned query embeddings (one per forecast step) decoded
    through a TransformerDecoderLayer with masked self-attention +
    cross-attention to encoder memory.  Output is Linear(d_model, 1)
    applied per token → 96 parallel predictions.

    No autoregressive loop.  No teacher forcing.
    ~94,945 params with d_model=64, 1 enc layer, 1 dec layer, dim_ff=144.
    """

    def __init__(self, n_features, d_model, n_heads, n_layers, dim_ff,
                 horizon, dropout=0.1):
        super().__init__()
        self.horizon = horizon

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        self.query_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers,
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, return_attn=False):
        """x: (B, lookback, n_features) → predictions: (B, horizon)"""
        B = x.size(0)

        h = self.input_proj(x)
        h = self.pos_enc(h)
        memory = self.transformer_encoder(h)

        queries = self.query_embed.expand(B, -1, -1)

        causal_mask = torch.triu(
            torch.full((self.horizon, self.horizon), float('-inf'),
                       device=x.device),
            diagonal=1,
        )

        dec_out = self.transformer_decoder(
            queries, memory, tgt_mask=causal_mask,
        )

        preds = self.output_proj(dec_out).squeeze(-1)

        if return_attn:
            return preds, None
        return preds

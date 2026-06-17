import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert(dim_out % num_heads == 0, "dim_out % num_heads != 0")

        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)

        self.dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

    def forward(self, inputs):
        batches, tokens_len, _ = inputs.shape

        # shape: (batches, tokens_len, d_out)
        keys = self.w_key(inputs)
        queries = self.w_query(inputs)
        values = self.w_value(inputs)

        keys = keys.view(batches, tokens_len, self.num_heads, self.head_dim)
        values = values.view(batches, tokens_len, self.num_heads, self.head_dim)
        queries = queries.view(batches, tokens_len, self.num_heads, self.head_dim)

        # shape: (batches, num_heads, tokens_len, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-1, -2)

        # causal attention
        # underscore after method = in-place operations
        causal_mask = self.causal_mask.bool()[:tokens_len, :tokens_len]
        attn_scores.masked_fill_(causal_mask, -torch.inf)
        
        scale = keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = attn_weights @ values

        # revert previous transpose
        # shape: (batches, tokens_len, num_heads, head_dim)
        context_vecs = context_vecs.transpose(1, 2)

        # revert initial split by combining num_heads and head_dim
        context_vecs = context_vecs.contiguous()\
            .view(batches, tokens_len, self.dim_out)

        # added for completeness, LLMs have this
        context_vecs = self.out_proj(context_vecs)

        return context_vecs

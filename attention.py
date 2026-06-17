import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length

        self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

    def forward(self, inputs):
        keys = self.w_key(inputs)
        queries = self.w_query(inputs)
        values = self.w_value(inputs)

        attn_scores = queries @ keys.transpose(-1, -2)

        # causal attention
        # underscore after method = in-place operations
        context_length = self.context_length
        bool_causal_mask = self.causal_mask.bool()[:context_length, :context_length]
        attn_scores.masked_fill_(bool_causal_mask, -torch.inf)
        
        scale = keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = attn_weights @ values
        return context_vecs

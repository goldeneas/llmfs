import torch
from torch import dropout, nn
from attention import MultiHeadAttention
from feed_forward import FeedForward

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        vocab_size = cfg["vocab_size"]
        embedding_len = cfg["embedding_dim"]
        context_length = cfg["context_length"]
        self.token_emb_layer = torch.nn.Embedding(vocab_size, embedding_len)
        self.position_emb_layer = torch.nn.Embedding(context_length, embedding_len)

        dropout_rate = cfg["dropout_rate"]
        self.dropout_layer = nn.Dropout(dropout_rate)

        transformers_len = cfg["transformers_len"]
        self.transformers = nn.Sequential(
                *[GPTTransformerBlock(cfg) for _ in range(transformers_len)]
        )

        self.normalization_layer = GPTLayerNorm(embedding_len)
        self.out_layer = nn.Linear(embedding_len, vocab_size, bias=False)

    def forward(self, input_tokens):
        _, context_size = input_tokens.shape
        token_embeddings = self.token_emb_layer(input_tokens)

        pos_input = torch.arange(context_size, device=input_tokens.device)
        position_embeddings = self.position_emb_layer(pos_input)

        x = token_embeddings + position_embeddings
        x = self.dropout_layer(x)
        x = self.normalization_layer.forward(x)

        logits = self.out_layer(x)
        return logits


class GPTTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        embedding_len = cfg["embedding_len"]
        context_len = cfg["context_len"]
        dropout = cfg["dropout"]
        num_heads = cfg["num_heads"]

        self.attention = MultiHeadAttention(dim_in=embedding_len,
                        dim_out=embedding_len,
                        context_len=context_len,
                        dropout=dropout,
                        num_heads=num_heads)
        self.feed_forward = FeedForward(cfg)
        self.norm1 = GPTLayerNorm(embedding_len)
        self.norm2 = GPTLayerNorm(embedding_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1.forward(x)
        x = self.attention.forward(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2.forward(x)
        x = self.feed_forward.forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPTLayerNorm(nn.Module):
    def __init__(self, embedding_len):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_len))
        self.shift = nn.Parameter(torch.zeros(embedding_len))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return normalized_x * self.scale + self.shift

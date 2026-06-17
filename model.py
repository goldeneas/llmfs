import torch
from torch import nn

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

    def forward(self, x):
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

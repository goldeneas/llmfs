import argparse
import tiktoken
import torch
from importlib.metadata import version
from torch.utils.data import DataLoader

from attention import MultiHeadAttention
from dataset import GPTDataset

parser = argparse.ArgumentParser(prog="LLMfs",
                                   description="Custom LLM implementation from scratch",
                                   epilog="Built with <3 by goldeneas")

parser.add_argument("input_file", type=str)
args = parser.parse_args()


text_path = args.input_file
raw_text = None

with open(text_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Using tiktoken v{version('tiktoken')}")
print(f"Read file {text_path}: {len(raw_text)} characters")

# Preprocessing
context_size = 4
stride = context_size
encoding_name = "gpt2"
batch_size = 8
shuffle = True
drop_last = True
num_workers = 0
torch_seed = 123
embedding_dim = 256

print("[PRE-PROCESSING]")
print(f"Using context size: {context_size}")
print(f"Using stride: {stride}")
print(f"Using encoding: {encoding_name}")
print(f"Using batch size: {batch_size}")
print(f"Using shuffle: {shuffle}")
print(f"Using drop last: {drop_last}")
print(f"Using num workers: {num_workers}")
print(f"Using torch seed: {torch_seed}")
print(f"Using embedding dim: {embedding_dim}")

tokenizer = tiktoken.get_encoding(encoding_name)
dataset = GPTDataset(raw_text, tokenizer, context_size, stride)
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last, 
                        num_workers=num_workers)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

torch.manual_seed(torch_seed)

# Pre-processing pipeline
# Get input text
# Process input text into tokens
# Process tokens into embeddings (random values at first) -> token_embeddings
# Add position encoding to token embeddings -> position_embeddings
# Add attention weights to position_embeddings -> context_vectors

vocab_size = tokenizer.max_token_value + 1
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
token_embeddings = token_embedding_layer(inputs)

position_embedding_layer = torch.nn.Embedding(context_size, embedding_dim)
position_embeddings = position_embedding_layer(torch.arange(context_size))

input_embeddings = token_embeddings + position_embeddings

# Attention!
dim_in = 256
dim_out = 256
context_length = 4
dropout = 0.5
num_heads = 2

print("[ATTENTION]")
print(f"Using dim_in: {dim_in}")
print(f"Using dim_out: {dim_out}")
print(f"Using context_length: {context_length}")
print(f"Using dropout: {dropout}")
print(f"Using num_heads: {num_heads}")

sa = MultiHeadAttention(dim_in=dim_in,
                        dim_out=dim_out,
                        context_length=context_length,
                        dropout=dropout,
                        num_heads=num_heads)

context_vecs = sa.forward(input_embeddings)

print(f"Context vecs: {context_vecs}")

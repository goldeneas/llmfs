import argparse
import tiktoken
import torch
from importlib.metadata import version
from torch.utils.data import DataLoader

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


context_size = 4
stride = context_size
encoding_name = "gpt2"
batch_size = 8
shuffle = True
drop_last = True
num_workers = 0
torch_seed = 123
embedding_dim = 256
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

vocab_size = tokenizer.max_token_value + 1
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
token_embeddings = token_embedding_layer(inputs)

position_embedding_layer = torch.nn.Embedding(context_size, embedding_dim)
position_embeddings = position_embedding_layer(torch.arange(context_size))

input_embeddings = token_embeddings + position_embeddings

# multiply inputs by its transposed version
attn_weights = input_embeddings @ input_embeddings.transpose(-1, -2)
attn_norm_weights = torch.softmax(attn_weights, dim=-1)
context_vecs = attn_norm_weights @ input_embeddings

print(f"Calculated, simplified context vectors: {context_vecs}")

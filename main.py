import argparse
import tiktoken
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
stride = 4
encoding_name = "gpt2"
batch_size = 4
shuffle = True
drop_last = True
num_workers = 0
print(f"Using context size: {context_size}")
print(f"Using stride: {stride}")
print(f"Using encoding: {encoding_name}")
print(f"Using batch size: {batch_size}")
print(f"Using shuffle: {shuffle}")
print(f"Using drop last: {drop_last}")
print(f"Using num workers: {num_workers}")

tokenizer = tiktoken.get_encoding(encoding_name)
dataset = GPTDataset(raw_text, tokenizer, context_size, stride)
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last, 
                        num_workers=num_workers)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

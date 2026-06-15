from torch.utils.data import Dataset
import torch

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range (0, len(token_ids) - context_size, stride):
            input_chunk = token_ids[i : i+context_size]
            target_chunk = token_ids[i+1 : i+1+context_size]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # returns total rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

import torch
from torch.utils.data import Dataset

class AdditionalModule(Dataset):
    def __init__(self, dim):
        self.dim = dim
        self.data = self.prepare_data()

    def __len__(self):
        return self.dim ** 2

    def one_hot(self, x):
        out = torch.zeros(self.dim+1, dtype=torch.float64)
        out[x] = 1.0
        return out

    def equals_tensor(self):
        out = torch.zeros(self.dim+1, dtype=torch.float64)
        out[-1] = 1.0
        return out

    def prepare_data(self):
        data = torch.zeros((self.dim, self.dim), dtype=torch.float64)
        for x in range(self.dim):
            for y in range(self.dim):
                data[y][x] = (x + y) % self.dim
        return data

    def __getitem__(self, idx):
        index_y = idx // self.dim
        index_x = idx % self.dim

        value = self.data[index_y][index_x].long()

        return (
            self.one_hot(value),
            self.one_hot(index_x),
            self.one_hot(index_y),
            self.equals_tensor(),

        )
import torch
from torch.utils.data import Dataset

class AdditionalModule(Dataset):
    def __init__(self, dim):
        self.dim = dim
        self.data = self.prepare_data()

    def __len__(self):
        return self.dim ** 2


    def prepare_data(self):
        data = torch.zeros((self.dim, self.dim), dtype=torch.int64)
        for x in range(self.dim):
            for y in range(self.dim):
                data[y][x] = (x + y) % self.dim
        return data

    def __getitem__(self, idx):
        index_y = idx // self.dim
        index_x = idx % self.dim

        value = self.data[index_y][index_x].long()

        final = torch.cat([
            torch.tensor(index_x, dtype=torch.int64).unsqueeze(0),
            torch.tensor(index_x, dtype=torch.int64).unsqueeze(0),
            torch.tensor(self.dim+1, dtype=torch.int64).unsqueeze(0),
        ], dim=0)

        return (
            torch.tensor(value, dtype=torch.int64).unsqueeze(0),
            final
        )
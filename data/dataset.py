import torch
from torch.utils.data import Dataset


class ModularAdditionDataset(Dataset):
    """
    Dataset for (x + y) mod p task
    Все возможные пары x, y от 0 до p-1
    """

    def __init__(self, p=97, equal_token=None):
        """
        Args:
            p: modulo value
            equal_token: token for '=', если None, то p
        """
        self.p = p
        self.equal_token = equal_token if equal_token is not None else p

        # Создаем матрицу всех возможных пар и их результатов
        self.x_coords = torch.arange(p).repeat(p)  # [0,1,2,...,p-1, 0,1,2,...]
        self.y_coords = torch.arange(p).repeat_interleave(p)  # [0,0,...,0, 1,1,...,1, ...]

        # Вычисляем результаты
        self.results = (self.x_coords * self.y_coords + self.x_coords + self.y_coords) % p

    def __len__(self):
        return self.p * self.p

    def __getitem__(self, idx):
        """
        Returns:
            input_seq: [x, y, =] - последовательность из 3 токенов
            target: результат (x + y) % p
        """
        x = self.x_coords[idx].long()
        y = self.y_coords[idx].long()
        result = self.results[idx].long()

        # Входная последовательность: [x, y, =]
        input_seq = torch.tensor([x, y, self.equal_token], dtype=torch.long)

        # Целевое значение
        target = torch.tensor([result], dtype=torch.long)

        return target, input_seq
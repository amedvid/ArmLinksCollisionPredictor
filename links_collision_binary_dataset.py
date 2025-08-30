import os
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset


class LinkCollisionsBinaryDataset(Dataset):
    """
    Multiple binary files, fixed record layout:
      record = [n_in float32 ... , n_out float32 ...]
    """

    def __init__(self, files, n_in=1854, n_out=8, dtype=np.float32):
        super().__init__()
        self.files = list(files)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.dtype = dtype
        self.rec_floats = self.n_in + self.n_out
        self.rec_bytes = self.rec_floats * np.dtype(dtype).itemsize

        # build memmaps and sizes
        self.mmaps = []
        self.sizes = []  # records per file
        for p in self.files:
            fsize = os.path.getsize(p)
            if fsize % self.rec_bytes != 0:
                raise ValueError(f"File size not multiple of record size: {p}")
            nrec = fsize // self.rec_bytes
            self.sizes.append(nrec)
            mm = np.memmap(p, mode="r", dtype=dtype, shape=(nrec, self.rec_floats))
            self.mmaps.append(mm)

        # prefix sums to map global idx -> (file_id, local_idx)
        self.prefix = np.cumsum([0, *self.sizes])  # len = len(files)+1
        self.total = int(self.prefix[-1])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total
        if not (0 <= idx < self.total):
            raise IndexError(idx)
        # find file: prefix[i] <= idx < prefix[i+1]
        fi = bisect.bisect_right(self.prefix, idx) - 1
        li = idx - self.prefix[fi]  # local index in file
        row = self.mmaps[fi][li]  # view [n_in + n_out]

        x_np = row[: self.n_in]
        y_np = row[self.n_in: self.n_in + self.n_out]
        # copy() to detach from memmap before torch
        x = torch.from_numpy(x_np.copy()).float()
        y = torch.from_numpy(y_np.copy()).float()
        return x, y

from torch.utils.data import Dataset
from torch.utils.data import Sampler
import numpy as np


class GroupedTimeSeriesDataset(Dataset):
    def __init__(self, data_X, data_Y, group_ids, split_ids):
        self.data_X = data_X
        self.data_Y = data_Y
        self.group_ids = group_ids
        self.split_ids = split_ids 

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_Y[idx], self.group_ids[idx], self.split_ids[idx]


class GroupedBatchSampler(Sampler):
    def __init__(self, group_ids, split_ids, batch_size, batch_shuffle):
        self.batch_shuffle = batch_shuffle
        self.group_ids = group_ids
        self.split_ids = split_ids
        self.batch_size = batch_size
        self.group_split_indices = self._get_group_split_indices()
        self.batches = self._create_batches()

    def _get_group_split_indices(self):
        """
        Create a dictionary mapping (group_id, split_id) to a list of indices that belong to that group and split.
        """
        group_split_indices = {}
        for idx, (group_id, split_id) in enumerate(zip(self.group_ids, self.split_ids)):
            key = (group_id, split_id)
            if key not in group_split_indices:
                group_split_indices[key] = []
            group_split_indices[key].append(idx)
        return group_split_indices

    def _create_batches(self):
        """
        Create batches of indices from group_split_indices.
        """
        batches = []
        for indices in self.group_split_indices.values():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                batches.append(batch)
        
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def shuffle(self):
        if self.batch_shuffle:
            np.random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)
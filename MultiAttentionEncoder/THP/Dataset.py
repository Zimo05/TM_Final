import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

# Keep imports anchored to this checkout.  The old absolute path pointed to a
# sibling copy of MultiAttentionEncoder and made CL runs silently mix files
# from two repositories.
_THP_DIR = str(Path(__file__).resolve().parent)
if _THP_DIR not in sys.path:
    sys.path.insert(0, _THP_DIR)
from TransformerModel import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.length = len(data)
        # store lengths for sorting
        self.seq_lengths = [len(t) for t in self.time]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


class LengthBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that groups sequences of similar length to minimize padding."""

    def __init__(self, data_source, batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seq_lengths = data_source.seq_lengths

    def __iter__(self):
        # Sort indices by sequence length
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            # Shuffle, then sort by length with some randomness for similar-length grouping
            np.random.shuffle(indices)
        indices = sorted(indices, key=lambda i: self.seq_lengths[i])

        # Create batches from sorted indices
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def get_dataloader(
        data, batch_size, shuffle=True, use_length_batch=True, num_workers=0):
    """ Prepare dataloader.

    Args:
        data: list of event sequences
        batch_size: batch size
        shuffle: whether to shuffle data
        use_length_batch: if True, group sequences by similar length to minimize padding
        num_workers: DataLoader worker processes. Keep 0 for notebook/macOS-safe use.
    """

    ds = EventData(data)

    if use_length_batch:
        sampler = LengthBatchSampler(ds, batch_size, shuffle=shuffle, drop_last=False)
        dl = torch.utils.data.DataLoader(
            ds,
            num_workers=num_workers,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )
    else:
        dl = torch.utils.data.DataLoader(
            ds,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
    return dl

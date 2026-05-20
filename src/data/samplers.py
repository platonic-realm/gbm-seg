"""Sampler that keeps same-file indices contiguous in the iteration.

The pre-refactor GBMDataset preloaded every (original + augmented variant)
volume into RAM at __init__, so __getitem__ was cheap and random shuffling
was free. The lazy-load refactor (d7aae92) made __getitem__ read the
relevant TIFF from disk per call — at which point random shuffling
becomes I/O-pathological: with N volumes and S samples per volume, a
fully random epoch causes ~N reads per worker per file (one per sample
hitting that file), instead of the obvious 1 read per worker per file.

`FileBlockRandomSampler` fixes this without giving up SGD-style randomness:

  * Shuffle the order of files each epoch (one cross-rank file order).
  * Shuffle the order of patches within each file (per-rank).
  * Yield each file's patches in a contiguous block before moving on.

Combined with a single-entry image cache around `_load_image` (the next
patch from the same file is a cache hit), each worker reads each of its
files exactly once per epoch.

The sampler also handles DDP: `num_replicas > 1` partitions FILES across
ranks (not flat indices, which would interleave files within a rank and
defeat the block structure). Per-rank epoch length is padded to a fixed
target so all ranks step in lockstep through the NCCL all-reduce barrier.
"""
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class FileBlockRandomSampler(Sampler[int]):
    """File-partitioned block-random sampler. Works for single-process
    (``num_replicas=1, rank=0``) and DDP (``num_replicas=world_size``).

    The dataset is expected to expose ``samples_per_image: list[int]``
    giving the number of samples per registered file, in the same order
    the dataset's flat-index space (``self.cumulative_sum``) is built
    from.

    Each epoch:
      1. Globally shuffle the file order using a shared per-epoch RNG
         (deterministic across ranks).
      2. Partition: rank k owns ``file_order[k::num_replicas]``.
      3. Per-rank RNG shuffles the patches inside each owned file.
      4. Yield each owned file's patches as a contiguous block.

    Per-rank length is padded to the maximum possible per-rank count
    (sum of the ``files_per_rank`` largest ``samples_per_image`` values)
    so every rank steps the same number of times — DDP NCCL needs that.
    Pad indices are repeats from the start of the rank's own sequence.
    """

    def __init__(self,
                 dataset,
                 num_replicas: int = 1,
                 rank: int = 0,
                 shuffle: bool = True,
                 seed: int = 0):
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"num_replicas must be >= 1 and 0 <= rank < num_replicas; "
                f"got num_replicas={num_replicas}, rank={rank}")
        self.samples_per_image: list[int] = list(dataset.samples_per_image)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Start-of-file flat-index offsets, parallel to samples_per_image.
        self._starts: list[int] = [0]
        for n in self.samples_per_image[:-1]:
            self._starts.append(self._starts[-1] + n)

        n_files = len(self.samples_per_image)
        self._files_per_rank = (n_files + num_replicas - 1) // num_replicas

        # Worst-case per-rank length: the rank gets the `files_per_rank`
        # largest files. We pad every rank to this number so length is
        # constant across ranks AND across epochs (required for the
        # DataLoader's len(), which the trainer's epoch loop uses).
        sorted_counts = sorted(self.samples_per_image, reverse=True)
        self._num_samples = sum(sorted_counts[:self._files_per_rank])

    def set_epoch(self, epoch: int) -> None:
        """Mirror DistributedSampler's API — the trainer calls this per
        epoch so the per-epoch shuffle is deterministic but varies.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        n_files = len(self.samples_per_image)

        # Shared per-epoch file shuffle (same on every rank).
        g_shared = torch.Generator()
        g_shared.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            file_order = torch.randperm(n_files, generator=g_shared).tolist()
        else:
            file_order = list(range(n_files))

        # Files this rank owns.
        my_files = file_order[self.rank::self.num_replicas]

        # Per-rank RNG for patch shuffling — different across ranks so two
        # ranks that happen to share a file (via the padding below) don't
        # iterate it identically.
        g_local = torch.Generator()
        g_local.manual_seed(self.seed + self.epoch * 31 + self.rank * 7919)

        seq: list[int] = []
        for fi in my_files:
            n_in_file = self.samples_per_image[fi]
            if self.shuffle:
                perm = torch.randperm(n_in_file, generator=g_local).tolist()
            else:
                perm = list(range(n_in_file))
            start = self._starts[fi]
            seq.extend(start + p for p in perm)

        # Pad with repeats of seq's prefix so every rank yields the same
        # number of indices per epoch. Truncate as a safety belt.
        if len(seq) < self._num_samples:
            # If seq is empty (impossible in practice — files_per_rank >= 1
            # and each file has >= 1 sample), this would loop forever, so
            # short-circuit.
            if not seq:
                return
            pad_needed = self._num_samples - len(seq)
            # Repeat seq cyclically until the pad is filled.
            while pad_needed > 0:
                take = min(len(seq), pad_needed)
                seq.extend(seq[:take])
                pad_needed -= take
        seq = seq[:self._num_samples]

        yield from seq

    def __len__(self) -> int:
        return self._num_samples

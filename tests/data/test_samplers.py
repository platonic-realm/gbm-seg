"""FileBlockRandomSampler: per-file contiguous blocks + DDP partitioning.

The lazy-load refactor in ds_train.py made random shuffling
I/O-pathological — every sample read its file from disk afresh. This
sampler groups indices by file so a single-entry image cache around
_load_image hits within a block. Tests below verify the block structure,
DDP partitioning, length stability, and that shuffle=False degrades
cleanly to file-ordered iteration.
"""
from collections import Counter

import pytest

from src.data.samplers import FileBlockRandomSampler


class _FakeDS:
    """Minimum surface the sampler needs from a dataset."""
    def __init__(self, samples_per_image):
        self.samples_per_image = list(samples_per_image)


def _file_for_index(idx, samples_per_image):
    """Reverse-lookup: given a flat index, return the file id it belongs to."""
    cum = 0
    for fi, n in enumerate(samples_per_image):
        cum += n
        if idx < cum:
            return fi
    raise ValueError(f"index {idx} out of range")


def _file_blocks(indices, samples_per_image):
    """Walk the sequence and return [(file_id, count), ...] runs."""
    blocks = []
    if not indices:
        return blocks
    cur_file = _file_for_index(indices[0], samples_per_image)
    n = 1
    for idx in indices[1:]:
        fi = _file_for_index(idx, samples_per_image)
        if fi == cur_file:
            n += 1
        else:
            blocks.append((cur_file, n))
            cur_file = fi
            n = 1
    blocks.append((cur_file, n))
    return blocks


def test_blocks_are_contiguous_single_process():
    """Same-file indices come out in one contiguous run per file."""
    spi = [3, 5, 2, 4]   # 4 files with 3/5/2/4 samples each
    ds = _FakeDS(spi)
    sampler = FileBlockRandomSampler(ds, num_replicas=1, rank=0, shuffle=True)
    sampler.set_epoch(0)
    indices = list(sampler)
    blocks = _file_blocks(indices, spi)
    # Each file id appears in exactly ONE block (no interleaving).
    seen = [fi for fi, _ in blocks]
    assert len(seen) == len(set(seen)), f"file ids re-appear (interleaving): {blocks}"


def test_all_patches_yielded_single_process():
    """Single-process: every (file, patch) is yielded exactly once."""
    spi = [3, 5, 2, 4]
    ds = _FakeDS(spi)
    sampler = FileBlockRandomSampler(ds, num_replicas=1, rank=0, shuffle=True)
    sampler.set_epoch(7)
    indices = list(sampler)
    assert sorted(indices) == list(range(sum(spi)))


def test_set_epoch_changes_file_order():
    """Different epochs should give different file orders (shuffle=True)."""
    spi = [4, 4, 4, 4, 4, 4]
    ds = _FakeDS(spi)
    sampler = FileBlockRandomSampler(ds, num_replicas=1, rank=0, shuffle=True)
    sampler.set_epoch(0)
    order_0 = [b[0] for b in _file_blocks(list(sampler), spi)]
    sampler.set_epoch(1)
    order_1 = [b[0] for b in _file_blocks(list(sampler), spi)]
    assert order_0 != order_1, "shuffle should differ across epochs"


def test_no_shuffle_is_file_ordered():
    """shuffle=False yields the natural file-ordered sequence."""
    spi = [3, 5, 2, 4]
    ds = _FakeDS(spi)
    sampler = FileBlockRandomSampler(ds, num_replicas=1, rank=0, shuffle=False)
    indices = list(sampler)
    assert indices == list(range(sum(spi)))


def test_ddp_partitions_files_across_ranks():
    """Under DDP each rank should see indices ONLY from its file partition."""
    spi = [4, 4, 4, 4, 4, 4, 4, 4]   # 8 files
    ds = _FakeDS(spi)
    seen_per_rank = []
    W = 4
    for r in range(W):
        s = FileBlockRandomSampler(ds, num_replicas=W, rank=r, shuffle=True)
        s.set_epoch(0)
        files_in_this_rank = {_file_for_index(i, spi) for i in s}
        seen_per_rank.append(files_in_this_rank)

    # Same file shuffle is shared across ranks (same seed + epoch), so
    # rank k owns file_order[k::W] → exactly 2 files per rank, disjoint.
    sizes = [len(s) for s in seen_per_rank]
    assert sizes == [2, 2, 2, 2], f"per-rank file counts: {sizes}"
    union = set().union(*seen_per_rank)
    assert union == set(range(8)), "every file should be owned by SOME rank"


def test_ddp_per_rank_length_is_identical():
    """All ranks emit the same number of indices per epoch (NCCL needs sync)."""
    # Uneven samples_per_image is the interesting case — without padding,
    # rank lengths would vary.
    spi = [10, 20, 30, 40, 50, 60, 70]   # 7 files of differing sizes
    ds = _FakeDS(spi)
    W = 3
    lens = []
    for r in range(W):
        s = FileBlockRandomSampler(ds, num_replicas=W, rank=r, shuffle=True)
        s.set_epoch(0)
        lens.append(len(list(s)))
    assert len(set(lens)) == 1, f"per-rank lengths diverge: {lens}"
    # __len__ should match the actual emit count.
    s0 = FileBlockRandomSampler(ds, num_replicas=W, rank=0, shuffle=True)
    assert len(s0) == lens[0]


def test_ddp_padded_indices_are_repeats_of_owned_files():
    """Padding to the per-rank target uses indices from the rank's OWN
    files (no cross-rank leakage), so each rank only ever reads its
    own assigned files in an epoch."""
    spi = [5, 5, 5, 5, 5, 100]   # one big file → padding pressure
    ds = _FakeDS(spi)
    W = 3
    for r in range(W):
        s = FileBlockRandomSampler(ds, num_replicas=W, rank=r, shuffle=True)
        s.set_epoch(0)
        files_seen = {_file_for_index(i, spi) for i in s}
        # File partition this rank owns: derived the same way as the sampler.
        # We don't reproduce the shuffled order here; we just check that the
        # files_seen size is at most files_per_rank — i.e., the padding
        # doesn't introduce files outside this rank's partition.
        files_per_rank = (len(spi) + W - 1) // W
        assert len(files_seen) <= files_per_rank, (
            f"rank {r} touches {len(files_seen)} files but should touch at "
            f"most {files_per_rank}: {files_seen}")


def test_invalid_rank_or_num_replicas_raises():
    ds = _FakeDS([3, 3, 3])
    with pytest.raises(ValueError):
        FileBlockRandomSampler(ds, num_replicas=0, rank=0)
    with pytest.raises(ValueError):
        FileBlockRandomSampler(ds, num_replicas=2, rank=-1)
    with pytest.raises(ValueError):
        FileBlockRandomSampler(ds, num_replicas=2, rank=2)

# Lazy loading and the block-random sampler

Each Z-upsampled training TIFF is ~5 GB. With offline augmentation,
there are 15 base TIFFs + 45 offline variants = **60 training volumes
× ~5 GB = ~300 GB**. We do not want to load all of that into RAM, and
we definitely don't want every DDP rank to load all of it.

This doc explains how the data pipeline handles this without crashing.

## The lazy-load dataset

`src/data/ds_train.py:GBMDataset` does **not** load TIFFs into RAM at
construction time. The constructor only:

1. Registers each training file path (original + every offline cache
   variant from the `methods_offline` list).
2. Stat-reads each file to get its shape (`image_shapes`) and number
   of patches (`samples_per_image`) — needed for `__len__()` and
   patch indexing.

When `__getitem__(i)` is called, it figures out which volume the index
`i` belongs to and lazy-loads *that one volume* through a
single-entry-per-worker cache (`_load_image` keeps the last-loaded
volume in memory and evicts on file change).

If two consecutive `__getitem__` calls hit the same file (likely
under the block-random sampler — see below) the second call hits the
cache and pays zero I/O. If they hit different files, the second
loads the new one and drops the old. Memory footprint per worker is
**one volume at a time**, ~5 GB.

This refactor landed in commit `d7aae92` (see git log for the full
story). Before it, the dataset preloaded every variant at constructor
time. With 60 variants × 5 GB × 4 dataloader workers × N DDP ranks,
the host machine swap-died.

## Why a single-entry cache is enough

The naïve choice would be an LRU cache big enough to hold a few
volumes. The single-entry cache is sufficient because the
**block-random sampler** guarantees that consecutive `__getitem__`
calls within a rank's batch tend to share a file.

## The FileBlockRandomSampler

`src/data/samplers.py:FileBlockRandomSampler` is a custom PyTorch
`Sampler`. It:

1. Shuffles the *order of files*.
2. Partitions files across DDP ranks (each rank gets a disjoint subset).
3. Yields each file's patches **contiguously** (every patch of file 1,
   then every patch of file 2, …).
4. Pads to a fixed length so NCCL collectives don't desync across ranks.

The key trade-off this samples induces:

- **Pro (huge):** Each `DataLoader` worker reads one file, then all
  its patches, then moves on. I/O is sequential and the worker-local
  cache hits constantly. Replacement reads happen only at file
  boundaries.
- **Con (smaller, manageable):** Within a single training step, the
  batch comes from one file (one rank) or a handful of files (across
  DDP ranks). That reduces gradient diversity per step — the model
  sees the same file repeatedly across consecutive steps before moving
  on. Mitigated by online augmentation and by the fact that 15
  volumes × 60+ patches each gives plenty of patches per file.

The block-random sampler is the **dominant reason the multi-node
training doesn't OOM**, but it also has subtle consequences for how
the model learns. In particular, see [phase coverage in the case
study](../07-case-study-z-jaggedness/diagnosis.md) — the sampler's
contiguous-per-file behaviour interacts with the Z-stride choice to
determine how many distinct **stacking phases** the model sees per
gradient window. That's the leverage we used in v3 (stride 2 covers
3 of 6 phases) and discussed for further interventions.

## Key code paths

- `src/data/ds_train.py:_register_file` — adds a file (original or
  offline variant) to `image_list` and `image_shapes`.
- `src/data/ds_train.py:_load_image` — the single-entry per-worker
  cache.
- `src/data/ds_train.py:_build_variant_cache` — invalidated when a
  worker resets at start-of-epoch.
- `src/data/samplers.py:FileBlockRandomSampler` — the custom sampler.
- `src/train/factory.py:createTrainDataset` — wires the sampler into
  the `DataLoader`.

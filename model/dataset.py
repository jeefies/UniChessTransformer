"""Fast memory-mapped dataset loader for 96-byte binary shards with vectorized numpy decoding."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from core.encoding import NUM_PLANES

RECORD_DTYPE = np.dtype([
    ("pawns",       "<u8"), ("knights", "<u8"), ("bishops", "<u8"),
    ("rooks",       "<u8"), ("queens",  "<u8"), ("kings",   "<u8"),
    ("occ_white",   "<u8"), ("occ_black", "<u8"),
    ("castling",    "u1"),  ("ep",   "u1"),     ("halfmove", "u1"),
    ("side",        "u1"),  ("rep",  "u1"),     ("promo",    "u1"),
    ("policy_move", "<u2", (5,)),
    ("policy_prob", "<u2", (5,)),
    ("wdl",         "<u2", (3,)),
])
assert RECORD_DTYPE.itemsize == 96, f"Record size must be 96 bytes, got {RECORD_DTYPE.itemsize}"

NO_EP = 255
NO_PROMO = 255
POLICY_SIZE = 4096
_SCALE = 65535.0

# Plane offsets matching core.encoding.py
_P_OWN, _P_OPP = 0, 6
_P_CASTLE, _P_EP, _P_HALF, _P_REP = 12, 16, 17, 18


def decode_batch(recs: np.ndarray) -> np.ndarray:
    """Vectorized batch decoding: records -> (N, 19, 8, 8) float32 numpy array.

    Pure numpy bitwise operations without python-chess overhead.
    """
    n = len(recs)
    planes = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)

    occ_w = recs["occ_white"].astype(np.uint64)
    occ_b = recs["occ_black"].astype(np.uint64)
    side = recs["side"].astype(bool)  # True = Black to move

    # Own vs opponent occupancy: swap if Black to move
    own = np.where(side, occ_b, occ_w)
    opp = np.where(side, occ_w, occ_b)

    bits = np.arange(64, dtype=np.uint64)
    for i, field in enumerate(("pawns", "knights", "bishops", "rooks", "queens", "kings")):
        bb = recs[field].astype(np.uint64)
        present = ((bb[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        is_own = ((own[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        is_opp = ((opp[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        planes[:, _P_OWN + i] = (present & is_own).astype(np.float32)
        planes[:, _P_OPP + i] = (present & is_opp).astype(np.float32)

    # Castling: low 4 bits are K Q k q
    c = recs["castling"].astype(np.uint8)
    own_k = np.where(side, (c >> 2) & 1, c & 1)
    own_q = np.where(side, (c >> 3) & 1, (c >> 1) & 1)
    opp_k = np.where(side, c & 1, (c >> 2) & 1)
    opp_q = np.where(side, (c >> 1) & 1, (c >> 3) & 1)
    for off, v in enumerate((own_k, own_q, opp_k, opp_q)):
        planes[:, _P_CASTLE + off] = v.astype(np.float32)[:, None]

    # En passant
    ep = recs["ep"].astype(np.int16)
    has_ep = ep != NO_EP
    rows = np.nonzero(has_ep)[0]
    if len(rows) > 0:
        planes[rows, _P_EP, ep[rows]] = 1.0

    # Halfmove clock and repetitions
    planes[:, _P_HALF] = (np.minimum(recs["halfmove"], 100) / 100.0).astype(np.float32)[:, None]
    planes[:, _P_REP] = (np.minimum(recs["rep"], 2) / 2.0).astype(np.float32)[:, None]

    planes = planes.reshape(n, NUM_PLANES, 8, 8)
    # If Black to move, flip vertically (ranks 1-8 flipped)
    flip = np.nonzero(side)[0]
    if len(flip) > 0:
        planes[flip] = planes[flip][:, :, ::-1, :]
    return planes


def decode_targets(recs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode target labels: (policy_target[N, 4096], promo_target[N], wdl_target[N, 3])."""
    n = len(recs)
    policy = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    mv = recs["policy_move"].astype(np.int32)
    pb = recs["policy_prob"].astype(np.float32) / _SCALE

    rows = np.repeat(np.arange(n), 5)
    np.add.at(policy, (rows, mv.ravel()), pb.ravel() * (pb.ravel() > 0))
    s = policy.sum(axis=1, keepdims=True)
    np.divide(policy, s, out=policy, where=s > 0)

    promo = recs["promo"].astype(np.int64)
    promo = np.where(promo == NO_PROMO, -100, promo)  # -100 = PyTorch ignore_index

    wdl = recs["wdl"].astype(np.float32) / _SCALE
    ws = wdl.sum(axis=1, keepdims=True)
    np.divide(wdl, ws, out=wdl, where=ws > 0)

    return policy, promo, wdl


class BatchShardDataset(Dataset):
    """Memory-mapped multi-shard dataset with vectorized batch-level gathering."""

    def __init__(self, shard_paths: Sequence[str | Path]):
        self.paths = [Path(p) for p in shard_paths]
        if not self.paths:
            raise FileNotFoundError("No shard paths provided")
        self._maps: list[np.memmap | None] = [None] * len(self.paths)
        counts = [p.stat().st_size // RECORD_DTYPE.itemsize for p in self.paths]
        self.offsets = np.cumsum([0] + counts)
        self.total = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.total

    def _map(self, i: int) -> np.memmap:
        if self._maps[i] is None:
            self._maps[i] = np.memmap(self.paths[i], dtype=RECORD_DTYPE, mode="r")
        return self._maps[i]

    def _gather(self, indices: np.ndarray) -> np.ndarray:
        shard = np.searchsorted(self.offsets, indices, side="right") - 1
        out = np.empty(len(indices), dtype=RECORD_DTYPE)
        for s in np.unique(shard):
            sel = shard == s
            local = indices[sel] - self.offsets[s]
            out[sel] = self._map(int(s))[local]
        return out

    def __getitem__(self, indices) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        idx = np.asarray(indices, dtype=np.int64)
        recs = self._gather(idx)

        x = torch.from_numpy(decode_batch(recs))
        p, pr, w = decode_targets(recs)
        return x, torch.from_numpy(p), torch.from_numpy(pr), torch.from_numpy(w)


class ShardDataset(Dataset):
    """Single-item indexing dataset for general access or slicing."""

    def __init__(self, shard_paths: Sequence[str | Path]):
        self.paths = [Path(p) for p in shard_paths]
        if not self.paths:
            raise FileNotFoundError("No shard paths provided")
        self._maps: list[np.memmap | None] = [None] * len(self.paths)
        counts = [p.stat().st_size // RECORD_DTYPE.itemsize for p in self.paths]
        self.offsets = np.cumsum([0] + counts)
        self.total = int(self.offsets[-1])

    def __len__(self) -> int:
        return self.total

    def _map(self, i: int) -> np.memmap:
        if self._maps[i] is None:
            self._maps[i] = np.memmap(self.paths[i], dtype=RECORD_DTYPE, mode="r")
        return self._maps[i]

    def __getitem__(self, idx: int):
        s = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        rec = self._map(s)[idx - self.offsets[s]:idx - self.offsets[s] + 1]
        x = decode_batch(rec)[0]
        p, pr, w = decode_targets(rec)
        return torch.from_numpy(x), torch.from_numpy(p[0]), torch.tensor(pr[0]), torch.from_numpy(w[0])


def get_shards(shard_dir: str | Path, pattern: str = "evals_*.bin") -> list[Path]:
    """Find and return all matching shard paths sorted."""
    paths = sorted(Path(shard_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards found in {shard_dir} matching {pattern}")
    return paths


def make_loader(
    shard_paths: Sequence[str | Path],
    batch_size: int,
    *,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> tuple[BatchShardDataset, DataLoader]:
    """Create a DataLoader configured for high-speed batched decoding."""
    ds = BatchShardDataset(shard_paths)
    base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=drop_last)
    loader = DataLoader(
        ds,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return ds, loader

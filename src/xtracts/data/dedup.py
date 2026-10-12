"""Exact + near-duplicate detection.

Two images are considered duplicates when either
  * their SHA-256 is identical (exact duplicate), or
  * BOTH perceptual hashes (pHash and dHash) are within a Hamming-distance
    threshold AND their aspect ratios are similar (near duplicate).

Grouping is KEEPER-ANCHORED (greedy), not transitive: images are visited from
best to worst (largest resolution, then largest file). The first unassigned
image becomes a keeper, and every still-unassigned image that is a direct
match of THAT keeper joins its group. An image never joins a group just
because it matches some other member, so long chains
(A~B, B~C, C~D ... => A and D grouped) cannot form.

Nothing is deleted from disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# popcount lookup for one byte
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hex_to_u64(series: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in series], dtype=np.uint64)


def _hamming_matrix(h: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances for an array of uint64 hashes -> (n, n) uint8."""
    x = np.bitwise_xor(h[:, None], h[None, :])          # (n, n) uint64
    return _POPCOUNT[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1).astype(np.uint8)


def greedy_assign(near: np.ndarray, order: list[int]) -> np.ndarray:
    """Keeper-anchored grouping.

    near  : (n, n) bool, near[i, j] True if i and j are direct matches
    order : visiting order, best keeper candidates first
    Returns an int array: the index of the keeper each image belongs to
    (a keeper maps to itself).
    """
    n = len(order)
    keeper_of = np.full(n, -1, dtype=int)
    for i in order:
        if keeper_of[i] != -1:
            continue
        keeper_of[i] = i
        keeper_of[near[i] & (keeper_of == -1)] = i
    return keeper_of


def find_duplicates(inv: pd.DataFrame, phash_threshold: int = 6, dhash_threshold: int = 6,
                    aspect_tolerance: float = 0.05) -> pd.DataFrame:
    """Annotate the inventory with duplicate-cluster information.

    Adds columns:
        cluster_id    int for images in a multi-image cluster, else -1
        cluster_size  number of images in that cluster (1 for uniques)
        dup_type      unique | keeper | exact | near
                      (exact/near = a removed copy and how it matched its keeper)
        is_keeper     True for the image kept from each cluster (and all uniques)
        keeper_path   path of the cluster's keeper (own path if unique/keeper)
        phash_dist    pHash distance to the keeper (NaN for uniques)
    Only rows with status == "ok" take part; others get is_keeper=False.
    """
    inv = inv.copy()
    inv["cluster_id"] = -1
    inv["cluster_size"] = 1
    inv["dup_type"] = "unique"
    inv["is_keeper"] = False
    inv["keeper_path"] = None
    inv["phash_dist"] = np.nan

    ok_idx = inv.index[inv["status"] == "ok"].to_numpy()
    if len(ok_idx) == 0:
        return inv
    ok = inv.loc[ok_idx]

    ph = _hamming_matrix(_hex_to_u64(ok["phash"]))
    dh = _hamming_matrix(_hex_to_u64(ok["dhash"]))
    ratio = (ok["width"] / ok["height"]).to_numpy()
    ratio_ok = np.abs(ratio[:, None] - ratio[None, :]) / ratio[:, None] <= aspect_tolerance
    shas = ok["sha256"].to_numpy()
    same_bytes = shas[:, None] == shas[None, :]
    near = same_bytes | ((ph <= phash_threshold) & (dh <= dhash_threshold) & ratio_ok)
    np.fill_diagonal(near, False)

    area = (ok["width"] * ok["height"]).to_numpy()
    size = ok["file_size"].to_numpy()
    paths = ok["path"].to_numpy()
    # best keeper first: biggest resolution, then biggest file, then alphabetical path
    order = sorted(range(len(ok)), key=lambda m: (-area[m], -size[m], paths[m]))
    keeper_of = greedy_assign(near, order)

    cluster_counter = 0
    for k in np.unique(keeper_of):
        members = np.nonzero(keeper_of == k)[0]
        if len(members) == 1:
            inv.loc[ok_idx[k], ["is_keeper", "keeper_path"]] = [True, paths[k]]
            continue
        for m in members:
            idx = ok_idx[m]
            inv.loc[idx, ["cluster_id", "cluster_size", "keeper_path"]] = [
                cluster_counter, len(members), paths[k]]
            if m == k:
                inv.loc[idx, ["is_keeper", "dup_type"]] = [True, "keeper"]
            else:
                inv.loc[idx, "dup_type"] = "exact" if shas[m] == shas[k] else "near"
                inv.loc[idx, "phash_dist"] = float(ph[m, k])
        cluster_counter += 1
    return inv


def summarize(inv: pd.DataFrame) -> dict:
    """Headline numbers for the dataset report."""
    status = inv["status"].value_counts().to_dict()
    dup = inv[inv["cluster_id"] >= 0]
    sizes = dup.groupby("cluster_id").size().sort_values(ascending=False)
    return {
        "total_files": len(inv),
        "status_counts": status,
        "valid_images": int((inv["status"] == "ok").sum()),
        "duplicate_clusters": int(dup["cluster_id"].nunique()),
        "exact_duplicates_removed": int((inv["dup_type"] == "exact").sum()),
        "near_duplicates_removed": int((inv["dup_type"] == "near").sum()),
        "unique_images_kept": int(inv["is_keeper"].sum()),
        "largest_cluster_sizes": [int(s) for s in sizes.head(5)],
    }

import dataclasses
from typing import Dict, List, Tuple

import numpy as np
import torch
from ogb.linkproppred import LinkPropPredDataset


@dataclasses.dataclass
class BioKGContext:
    dataset: LinkPropPredDataset
    split_edge: Dict[str, Dict[str, torch.Tensor]]
    type_names: List[str]
    type_offsets: Dict[str, int]
    num_nodes_by_type: Dict[str, int]
    num_entities: int
    num_relations: int


def load_biokg() -> BioKGContext:
    dataset = LinkPropPredDataset(name="ogbl-biokg")
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    type_names = sorted(list(data["num_nodes_dict"].keys()))
    type_offsets: Dict[str, int] = {}
    cursor = 0
    for t in type_names:
        type_offsets[t] = cursor
        cursor += int(data["num_nodes_dict"][t])

    num_relations = int(split_edge["train"]["relation"].max().item() + 1)

    return BioKGContext(
        dataset=dataset,
        split_edge=split_edge,
        type_names=type_names,
        type_offsets=type_offsets,
        num_nodes_by_type={k: int(v) for k, v in data["num_nodes_dict"].items()},
        num_entities=cursor,
        num_relations=num_relations,
    )


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def to_global_ids(local_ids: np.ndarray, type_names: np.ndarray, offsets: Dict[str, int]) -> np.ndarray:
    local_ids = _to_numpy(local_ids)
    type_names = _to_numpy(type_names)

    if type_names.shape != local_ids.shape:
        type_names = np.broadcast_to(type_names, local_ids.shape)

    flat_local = local_ids.reshape(-1).astype(np.int64, copy=False)
    flat_type = type_names.reshape(-1)
    out = np.empty_like(flat_local, dtype=np.int64)

    norm_type = np.empty(flat_type.shape[0], dtype=object)
    for i, t in enumerate(flat_type):
        if isinstance(t, bytes):
            norm_type[i] = t.decode("utf-8")
        else:
            norm_type[i] = str(t)

    unique_types = np.unique(norm_type)
    for t in unique_types.tolist():
        mask = norm_type == t
        out[mask] = flat_local[mask] + int(offsets[t])

    return out.reshape(local_ids.shape)


def split_to_global(split: Dict[str, torch.Tensor], offsets: Dict[str, int]) -> Dict[str, np.ndarray]:
    head = _to_numpy(split["head"]).astype(np.int64)
    tail = _to_numpy(split["tail"]).astype(np.int64)
    relation = _to_numpy(split["relation"]).astype(np.int64)
    head_type = _to_numpy(split["head_type"])
    tail_type = _to_numpy(split["tail_type"])

    out = {
        "head": to_global_ids(head, head_type, offsets),
        "tail": to_global_ids(tail, tail_type, offsets),
        "relation": relation,
        "head_type": head_type,
        "tail_type": tail_type,
    }

    if "head_neg" in split:
        out["head_neg"] = to_global_ids(_to_numpy(split["head_neg"]), np.expand_dims(head_type, 1), offsets)
    if "tail_neg" in split:
        out["tail_neg"] = to_global_ids(_to_numpy(split["tail_neg"]), np.expand_dims(tail_type, 1), offsets)
    return out


def type_id_buckets(offsets: Dict[str, int], num_nodes_by_type: Dict[str, int]) -> Dict[str, np.ndarray]:
    buckets: Dict[str, np.ndarray] = {}
    for t, offset in offsets.items():
        n = num_nodes_by_type[t]
        buckets[t] = np.arange(offset, offset + n, dtype=np.int64)
    return buckets


def group_indices_by_relation(relations: np.ndarray) -> Dict[int, np.ndarray]:
    rel_to_idx: Dict[int, List[int]] = {}
    for i, r in enumerate(relations.tolist()):
        rel_to_idx.setdefault(int(r), []).append(i)
    return {r: np.array(idxs, dtype=np.int64) for r, idxs in rel_to_idx.items()}

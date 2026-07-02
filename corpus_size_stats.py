#!/usr/bin/env python3
"""
Report maximum and average tensor input sizes for each API in a corpus directory.

For each API subdirectory that contains abstract_inputs.jsonl, the script reads
every abstract input record and computes the total tensor memory (bytes) across
all tensor parameters in that record, using the stored shapes and dtype indices.

Only parameters whose type is "tensor" or "tensor_list" in signatures.json are
counted; non-tensor parameters (integer, float, list, tuple, boolean, …) are
stored in the same [value, [dtype], [range]] format by get_ll and would
otherwise be misidentified as tensors.

Usage:
    python corpus_size_stats.py <corpus_dir> [signatures_json]

    signatures_json defaults to signatures.json in the project root.

Output columns:
    n           - number of abstract inputs
    avg (MB)    - average total tensor bytes per input, in MB
    max (MB)    - maximum total tensor bytes in any single input
    avg elems   - average total number of tensor elements per input
    max elems   - maximum total number of tensor elements in any single input
"""

import json
import math
import sys
from pathlib import Path

# Dtype index → item size in bytes, matching utils/defaults.py list_of_available_dtypes:
# [bool, np.int8, np.int16, np.int32, np.int64, np.uint8,
#  np.float16, np.float32, np.float64, np.complex64, np.complex128, str, np.dtype]
_DTYPE_ITEMSIZE = [
    1,   # 0  bool
    1,   # 1  np.int8
    2,   # 2  np.int16
    4,   # 3  np.int32
    8,   # 4  np.int64
    1,   # 5  np.uint8
    2,   # 6  np.float16
    4,   # 7  np.float32
    8,   # 8  np.float64
    8,   # 9  np.complex64
    16,  # 10 np.complex128
    0,   # 11 str  (no fixed size)
    0,   # 12 np.dtype (no fixed size)
]

_TENSOR_TYPES = {"tensor", "tensor_list"}


def _load_signatures(sig_path: Path) -> dict:
    """Return {api_name: set_of_tensor_param_names}."""
    raw = json.loads(sig_path.read_text())
    result = {}
    for api, sig in raw.items():
        tensor_params = set()
        for section in ("args", "kwargs"):
            for param, ptype in sig.get(section, {}).items():
                if ptype in _TENSOR_TYPES:
                    tensor_params.add(param)
        inner = sig.get("inner", {})
        for section in ("args", "kwargs"):
            for param, ptype in inner.get(section, {}).items():
                if ptype in _TENSOR_TYPES:
                    tensor_params.add(param)
        result[api] = tensor_params
    return result


def _dtype_idx(v) -> int:
    """Extract dtype index — stored as bare int (model_to_abs) or [int] (get_ll)."""
    return v[1][0] if isinstance(v[1], list) else v[1]


def _tensor_stats(v):
    """Return (n_elements, n_bytes) for one tensor abstract value."""
    shape = v[0]
    dtype_idx = _dtype_idx(v)
    n_elems = math.prod(shape) if shape else 1
    itemsize = _DTYPE_ITEMSIZE[dtype_idx] if 0 <= dtype_idx < len(_DTYPE_ITEMSIZE) else 0
    return n_elems, n_elems * itemsize


def _record_stats(record: dict, tensor_params: set):
    """Return (total_elements, total_bytes) summed over tensor parameters only."""
    total_elems = total_bytes = 0
    for param, v in record.items():
        if param not in tensor_params:
            continue
        if not (isinstance(v, list) and len(v) >= 1 and isinstance(v[0], list)):
            continue
        elems, nbytes = _tensor_stats(v)
        total_elems += elems
        total_bytes += nbytes
    return total_elems, total_bytes


def _api_key(dir_name: str) -> str:
    """Map corpus directory name (e.g. torch.nn.AvgPool2d_5) to signatures key."""
    return dir_name


def analyse_api(jsonl_path: Path, tensor_params: set):
    """Return list of (total_elements, total_bytes) per record."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            results.append(_record_stats(record, tensor_params))
    return results


def main(corpus_dir: str, sig_path: str = "signatures.json"):
    corpus = Path(corpus_dir)
    if not corpus.is_dir():
        print(f"error: {corpus_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    sig_file = Path(sig_path)
    if not sig_file.exists():
        print(f"error: signatures file not found: {sig_path}", file=sys.stderr)
        sys.exit(1)

    signatures = _load_signatures(sig_file)

    api_dirs = sorted(
        d for d in corpus.iterdir()
        if d.is_dir() and (d / "abstract_inputs.jsonl").exists()
    )

    if not api_dirs:
        print(f"no APIs with abstract_inputs.jsonl found under {corpus_dir}")
        return

    col = max(len(d.name) for d in api_dirs)
    hdr = (f"{'API':<{col}}  {'n':>6}  "
           f"{'avg (MB)':>10}  {'max (MB)':>10}  "
           f"{'avg elems':>12}  {'max elems':>12}")
    print(hdr)
    print("-" * len(hdr))

    for api_dir in api_dirs:
        api_key = _api_key(api_dir.name)
        tensor_params = signatures.get(api_key, set())
        if not tensor_params:
            print(f"{api_dir.name:<{col}}  (no tensor params in signature or unknown API)")
            continue

        stats = analyse_api(api_dir / "abstract_inputs.jsonl", tensor_params)
        if not stats:
            continue

        n = len(stats)
        all_elems = [e for e, _ in stats]
        all_bytes = [b for _, b in stats]

        avg_mb = (sum(all_bytes) / n) / (1024 ** 2)
        max_mb = max(all_bytes) / (1024 ** 2)
        avg_el = sum(all_elems) / n
        max_el = max(all_elems)

        print(f"{api_dir.name:<{col}}  {n:>6}  "
              f"{avg_mb:>10.4f}  {max_mb:>10.4f}  "
              f"{avg_el:>12,.0f}  {max_el:>12,}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"usage: {Path(sys.argv[0]).name} <corpus_dir> [signatures_json]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "signatures.json")

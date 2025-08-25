"""
Script to create multiple reduced variants of a .vbrc file by removing different percentages
of elements from csr_val while maintaining the CSR structure.

This script will read from a hardcoded .vbrc file and generate variants in the same directory.
"""

import os
import random
import re
from typing import List, Tuple, Type, TypeVar, Callable


T = TypeVar("T", int, float)


def parse_array(line: str, key: str, caster: Callable[[str], T]) -> Tuple[str, List[T]]:
    """Parse a line like 'key=[a,b,c,...]' and return (prefix, typed list)."""
    assert line.startswith(key + "=") or line.startswith(key + "=["), f"Line does not start with {key}="
    prefix, arr_str = line.split("=", 1)
    assert prefix.strip() == key, f"Unexpected key in line: {prefix}"
    assert arr_str.strip().startswith("["), "Array must start with '['"
    assert arr_str.strip().endswith("]"), "Array must end with ']'"
    body = arr_str.strip()[1:-1]
    # Safe parse by splitting on commas and converting
    parts = [] if body.strip() == "" else [p.strip() for p in body.split(",")]
    values: List[T] = []
    for p in parts:
        if p == "":
            continue
        values.append(caster(p))
    return prefix, values


def format_array(prefix: str, values: List[int] | List[float]) -> str:
    if all(isinstance(v, int) for v in values):
        content = ",".join(str(int(v)) for v in values)
    else:
        content = ",".join(repr(float(v)) for v in values)
    return f"{prefix}=[{content}]"


def load_vbrc_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_vbrc_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def find_line_index(lines: List[str], key: str) -> int:
    for i, line in enumerate(lines):
        if line.startswith(key + "=") or line.startswith(key + "=["):
            return i
    raise ValueError(f"Key {key} not found in file")


def sampled_removal_mask(indptr: List[int], drop_fraction: float, seed: int | None) -> List[bool]:
    """Return a mask for entries in the CSR data (length == indptr[-1]) marking which
    entries to keep (True) or drop (False). We drop approximately drop_fraction of entries.

    We sample uniformly at random over all nonzero positions, but maintain row structure:
    each row keeps a subset of its entries, and rows are allowed to become empty.
    """
    if not indptr:
        return []
    nnz = indptr[-1]
    num_drop = int(nnz * drop_fraction)
    num_drop = max(0, min(num_drop, nnz))
    rng = random.Random(seed)
    to_drop = set(rng.sample(range(nnz), num_drop)) if num_drop > 0 else set()
    mask = [i not in to_drop for i in range(nnz)]
    return mask


def apply_mask_to_csr(csr_val: List[float], indices: List[int], indptr: List[int], mask: List[bool]) -> Tuple[List[float], List[int], List[int]]:
    assert len(csr_val) == len(indices) == len(mask)
    n_rows = len(indptr) - 1
    new_val: List[float] = []
    new_indices: List[int] = []
    new_indptr: List[int] = [0]
    cursor = 0
    for r in range(n_rows):
        row_start = indptr[r]
        row_end = indptr[r + 1]
        kept_in_row = 0
        for i in range(row_start, row_end):
            if mask[i]:
                new_val.append(float(csr_val[i]))
                new_indices.append(int(indices[i]))
                kept_in_row += 1
        new_indptr.append(new_indptr[-1] + kept_in_row)
        cursor = row_end
    assert new_indptr[-1] == len(new_val) == len(new_indices)
    return new_val, new_indices, new_indptr


def main():
    # Hardcoded input file path - change this to your .vbrc file
    input_file = "/local/scratch/a/das160/SABLE/Generated_VBR_split/brainpc2/brainpc2.vbrc"  # Change this to your actual .vbrc file name
    
    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(input_file))
    if not input_dir:
        input_dir = "."  # Current directory if no path specified
    
    # Create output directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    lines = load_vbrc_lines(input_file)

    # Locate lines
    idx_val = find_line_index(lines, "csr_val")
    idx_indices = find_line_index(lines, "indices")
    idx_indptr = find_line_index(lines, "indptr")

    # Parse arrays
    _, csr_val = parse_array(lines[idx_val], "csr_val", float)
    _, indices = parse_array(lines[idx_indices], "indices", int)
    _, indptr = parse_array(lines[idx_indptr], "indptr", int)

    print(f"Original file: {input_file}")
    print(f"Original nnz: {len(csr_val)}")
    print(f"Creating variants in: {input_dir}")
    print()

    # Create variants for each percentage
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        fraction = pct / 100.0
        
        # Build mask and apply
        mask = sampled_removal_mask(indptr, fraction, 42) # Hardcoded seed for reproducibility
        new_val, new_indices, new_indptr = apply_mask_to_csr(csr_val, indices, indptr, mask)

        # Create output filename
        output_filename = f"{base_name}_reduced_{pct}pct.vbrc"
        output_path = os.path.join(input_dir, output_filename)

        # Create a copy of lines for this variant
        variant_lines = lines.copy()
        
        # Replace lines; keep original formatting/order otherwise
        variant_lines[idx_val] = format_array("csr_val", new_val)
        variant_lines[idx_indices] = format_array("indices", new_indices)
        variant_lines[idx_indptr] = format_array("indptr", new_indptr)

        write_vbrc_lines(output_path, variant_lines)
        
        reduction = len(csr_val) - len(new_val)
        reduction_pct = (reduction / len(csr_val)) * 100
        
        print(f"  {pct:2d}% removal: {output_filename}")
        print(f"    nnz: {len(csr_val)} -> {len(new_val)} (removed {reduction}, actual reduction: {reduction_pct:.1f}%)")
    
    print(f"\nAll variants created successfully in {input_dir}")


if __name__ == "__main__":
    main()



import argparse
import numpy as np
import glob
import torch
import h5py
from functools import partial
from typing import Any
from ..functions.hw import HasegawaWatakini

def test_file(file_path: str, batch_size: int, verbose: bool) -> None:
    with h5py.File(file_path, "r") as h5_file:
        parameters = dict(h5_file.attrs)
        L = 2 * torch.pi / parameters["k0"]
        dx = L / parameters["x_save"]
        steps = len(h5_file["density"])
        
        if verbose:
            print(f"\nTesting file: {file_path}")
            print(f"Total steps (time slices): {steps}")
        
        # Instantiate the HasegawaWatakini object once per file.
        hw_instance = HasegawaWatakini()
        
        for i in range(0, steps, batch_size):
            n = h5_file["density"][i : i + batch_size]
            p = h5_file["phi"][i : i + batch_size]
            o = h5_file["omega"][i : i + batch_size]
            
            # Map property names to (calculation function, expected data)
            property_mapping = {
                "gamma_n": (
                    partial(hw_instance.get_gamma_n, n=n, p=p, dx=dx, dtype='numpy'),
                    h5_file["gamma_n"][i : i + batch_size]
                )
                # Additional properties can be added here.
            }
            
            for name, (calc_func, expected) in property_mapping.items():
                calculated = calc_func()
                if verbose:
                    print(f"\nBatch {i} to {i + batch_size} for '{name}':")
                    print(f"Calculated: {calculated}")
                    print(f"Expected:   {expected}")
                
                if not np.allclose(calculated, expected):
                    raise AssertionError(
                        f"Mismatch in {name} for file {file_path} at batch starting at {i}"
                    )
                elif verbose:
                    print(f"'{name}' passed for batch starting at {i}.")

def main(args: Any) -> None:
    file_paths = glob.glob(f"{args.input_dir}/*.h5")
    if args.verbose:
        print(f"Found {len(file_paths)} test file(s) in '{args.input_dir}'.")
    if not file_paths:
        raise FileNotFoundError("No .h5 files found in the specified directory.")
    
    for file_path in file_paths:
        test_file(file_path, args.batch_size, args.verbose)
    
    print("\nAll tests passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test derived quantities prediction against dataset values"
    )
    parser.add_argument(
        "--input_dir", "-i", required=True, type=str,
        help="Directory containing .h5 test files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output for detailed test progress"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1,
        help="Batch size for processing tensors (default: 1)"
    )
    args = parser.parse_args()
    
    try:
        main(args)
    except AssertionError as e:
        print("Test failed:", e)
        exit(1)
    except Exception as e:
        print("Error encountered:", e)
        exit(1)

import argparse
from functools import partial
import torch
import h5py
from typing import String, List, Union, Tuple
import glob
from ..functions.hw import HasegawaWatakini

def main(args):
    # Load h5 files to open
    files_paths = glob.glob(args.input_dir + "/*.h5")
    if args.verbose:
        print(f"Found {len(files_paths)} test files.")
    
    for file_path in files_paths:
        with h5py.File(file_path, "r+") as h5_file:
            parameters = dict(h5_file.attrs)
            L = 2 * torch.pi / parameters["k0"]
            dx = L / parameters["x_save"]
            c1 = parameters["c1"]
            dt = parameters["frame_dt"]
            steps = len(h5_file["density"])
            for i in range(0,i, args.batch_size):
                n = h5_file["density"][i : i + args.batch_size]
                p = h5_file["phi"][i : i + args.batch_size]
                o = h5_file["omega"][i : i + args.batch_size]

                property_mapping = {
                "gamma_n": (partial(HasegawaWatakini.get_gamma_n, n=n, p=p, dx=dx),
                            h5_file["gamma_n"][i : i + args.batch_size])
                # TODO - Implement these
                # "gamma_n_spectral": partial(get_gamma_n_spectrally, n=n, p=p, dx=dx),
                # "gamma_c": partial(get_gamma_c, n=n, p=p, c1=c1, dx=dx),
                # "energy": partial(get_energy, n=n, phi=p, dx=dx),
                # "thermal_energy": partial(get_energy_N_spectrally, n=n),
                # "kinetic_energy": partial(get_energy_V_spectrally, p=p, dx=dx),
                # "enstrophy": partial(get_enstrophy, n=n, omega=o, dx=dx),
                # "enstrophy_phi": partial(get_enstrophy_phi, n=n, phi=p, dx=dx),
                }
                for name, data in property_mapping:
                    if not (torch.isclose(data[0], data[1]).all):
                        raise ValueError(f"{name} found in dataset is not the same as calculated values")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", "-i", type=str, help="Directory in which to look for h5 files to test")
    argparser.add_argument("--verbose", "-v", type=bool, help="Whether or not to output stuff")
    argparser.add_argument("--batch_size", "-b", type=int, help="Size in which to process tensors")

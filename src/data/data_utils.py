import os
import glob
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple

class H5PairDataset(Dataset):
    def __init__(self, input_dir, target_dir=None, device=torch.device("cpu"), derived=False):
        """
        Args:
            input_dir (str): Directory with input HDF5 files (e.g., input00.h5, input01.h5, ...)
            target_dir (str, optional): Directory with target HDF5 files (e.g., target00.h5, target01.h5, ...).
                                        If not provided, assumes both input and target files reside in input_dir.
            device (torch.device): Torch device to load tensors onto.
            derived (bool): Whether to load the derived gamma_n quantity.
        """
        self.input_dir = input_dir
        self.target_dir = target_dir if target_dir is not None else input_dir
        self.device = device
        self.derived = derived
        
        # Collect input and target files
        self.input_files = sorted(glob.glob(os.path.join(self.input_dir, 'input*.h5')))
        self.target_files = sorted(glob.glob(os.path.join(self.target_dir, 'target*.h5')))

        if len(self.input_files) != len(self.target_files):
            raise ValueError("Number of input files does not match number of target files.")

        # Initialize dx by reading the first file
        self._dx = self._read_dx()

    def __len__(self):
        return len(self.input_files)
    
    @property
    def dx(self):
        return self._dx

    def _read_dx(self):
        """Reads the dx attribute from the first HDF5 file."""
        if not self.input_files:
            raise ValueError("No input files found in the dataset directory.")
        with h5py.File(self.input_files[0], 'r') as f:
            return f.attrs['dx']

    def load_data(self, file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the HDF5 file and extracts relevant data."""
        with h5py.File(file, 'r') as f:
            density = f['density'][:]
            omega = f['omega'][:]
            phi = f['phi'][:]
            gamma_n = f['gamma_n'][:]

            # Stack input variables along the channel dimension
            data = np.stack((density, omega, phi), axis=1)  # (T, C, H, W)
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add batch dim
            
            return data_tensor.permute(0, 1, 2, 3, 4).unsqueeze(1).to(self.device), torch.tensor(gamma_n, dtype=torch.float32).to(self.device)

    def __getitem__(self, idx):
        """Returns a single data point (x, y) or (x, y, gamma_n) if derived is True."""
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]

        x, gamma_n_x = self.load_data(input_file)
        y, gamma_n_y = self.load_data(target_file)

        if self.derived:
            return x, y, gamma_n_y  # Ensure gamma_n is from the target file
        return (x[0], gamma_n_x), (y[0], gamma_n_y)

def get_data_loader(input_dir, target_dir=None, batch_size=32, shuffle=True, num_workers=4, 
                    device=torch.device("cpu"), derived=False):
    """
    Utility function to create a DataLoader for training.

    Args:
        input_dir (str): Directory containing input HDF5 files.
        target_dir (str, optional): Directory containing target HDF5 files. Defaults to None.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.
        device (torch.device): Torch device to load tensors onto.
        derived (bool): Whether to load the derived gamma_n quantity.

    Returns:
        DataLoader: A PyTorch DataLoader for the training dataset.
        H5PairDataset: The dataset instance (dx can be accessed via dataset.dx).
    """
    dataset = H5PairDataset(input_dir, target_dir, device, derived)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, dataset

def get_test_loader(test_input_dir, test_target_dir=None, batch_size=32, shuffle=False, num_workers=4,
                    device=torch.device("cpu"), derived=False):
    """
    Utility function to create a DataLoader for testing/evaluation.

    Args:
        test_input_dir (str): Directory containing test input HDF5 files.
        test_target_dir (str, optional): Directory containing test target HDF5 files.
                                         If not provided, assumes both test input and target files reside in test_input_dir.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data. Defaults to False for test loader.
        num_workers (int): Number of subprocesses to use for data loading.
        device (torch.device): Torch device to load tensors onto.
        derived (bool): Whether to load the derived gamma_n quantity.

    Returns:
        DataLoader: A PyTorch DataLoader for the test dataset.
        H5PairDataset: The dataset instance (dx can be accessed via dataset.dx).
    """
    dataset = H5PairDataset(test_input_dir, test_target_dir, device, derived)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, dataset

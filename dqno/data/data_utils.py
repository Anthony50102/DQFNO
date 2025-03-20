import os
import glob
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Any
import random

class PushForwardDataSet(Dataset):
    """
    Dataset class for loading HDF5 files containing simulation data.
    Expects each file to have the datasets: 'density', 'omega', 'phi', and 'gamma_n'.
    
    Files are opened using context managers to guarantee that file handles are closed
    immediately after reading.
    """

    def __init__(self, directory: str, pf_steps: int, chunk_size: int):
        self.files = sorted(glob.glob(os.path.join(directory, '*.h5')))
        random.shuffle(self.files)
        self.pf_steps = pf_steps + 1 # TODO - Fix this wierdness
        self.chunk_size = chunk_size

    @property
    def files(self) -> list:
        return self._files

    @files.setter
    def files(self, file_list: list):
        self._files = file_list

    def __len__(self) -> int:
        return len(self.files)

    def _get_data_dict(self, f: h5py.File) -> Dict[str, Any]:
        """
        Returns a dictionary with the full data loaded from an open h5py.File.
        This loads the data into memory so that the file can be closed safely.
        """
        return {
            'density': f['density'][()],
            'omega': f['omega'][()],
            'phi': f['phi'][()],
            'gamma_n': f['gamma_n'][()]
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Loads the HDF5 file at the specified index and returns a dictionary of numpy arrays.
        The file is opened and closed automatically.
        """
        file_path = self.files[index]
        with h5py.File(file_path, 'r') as f:
            data_dict = self._get_data_dict(f)
        return data_dict

    def get_chunk(self, file_path: str, custom_pf_step:int = None, rand_pf_step:bool = False) -> torch.Tensor:
        """
        For the provided file, pf_steps, and chunk_size,
        returns a tensor of indices for chunked data extraction.

        This method opens the file only to calculate the number of timesteps,
        then closes it immediately.
        """
        pf_steps = self.pf_steps
        if custom_pf_step:
            pf_steps = custom_pf_step
        if rand_pf_step:
            pf_steps = random.randint(1, pf_steps)

        print(pf_steps)
        with h5py.File(file_path, 'r') as f:
            data_timesteps = len(f['gamma_n'])

        total_timesteps = self.pf_steps * self.chunk_size
        diff = data_timesteps - total_timesteps
        # Randomly choose an end_index ensuring room for a complete chunk.
        end_index = random.randint(self.pf_steps * self.chunk_size, diff)
        start_index = end_index - self.pf_steps * self.chunk_size
        return torch.linspace(start_index, end_index - self.chunk_size, self.pf_steps).int()

    def get_data(self, file_path: str, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a chunk of data from the file starting at start_idx with the given chunk_size.
        The file is automatically closed after reading.
        
        Returns:
            state: Tensor concatenating density, omega, and phi (shape: [3, chunk_size]).
            gamma_n: Tensor for gamma_n (shape: [1, chunk_size]).
        """
        with h5py.File(file_path, 'r') as f:
            density = torch.from_numpy(f['density'][start_idx:start_idx + self.chunk_size]).unsqueeze(0)
            omega = torch.from_numpy(f['omega'][start_idx:start_idx + self.chunk_size]).unsqueeze(0)
            phi = torch.from_numpy(f['phi'][start_idx:start_idx + self.chunk_size]).unsqueeze(0)
            gamma_n = torch.from_numpy(f['gamma_n'][start_idx:start_idx + self.chunk_size]).unsqueeze(0) # (B, T)
        state = torch.cat((density, omega, phi)).unsqueeze(0).unsqueeze(0).permute(0,1,3,2,4,5) # (B,C,T,V,X,Y)
        return state, gamma_n


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
        self._dx, self.chunk_size = self._read()

    def __len__(self):
        return len(self.input_files)
    
    @property
    def dx(self):
        return self._dx

    def _read(self):
        """Reads the dx attribute from the first HDF5 file."""
        if not self.input_files:
            raise ValueError("No input files found in the dataset directory.")
        with h5py.File(self.input_files[0], 'r') as f:
            return f.attrs['dx'], len(f['gamma_n'])

    def load_data(self, file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the HDF5 file and extracts relevant data."""
        with h5py.File(file, 'r') as f:
            density = f['density'][:]
            omega = f['omega'][:]
            phi = f['phi'][:]
            gamma_n = f['gamma_n'][:]

            if self.chunk_size == None:
                self.chunk_size = len(gamma_n)

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

def push_forward(total_epoch: int ,epoch: int, type: str) -> bool:
    '''
    Returns true when we are doing a epoch of push forward testing is not then false
    '''
    if type == "only":
        print('#'*80)
        print('Push Forward Epoch')
        print('#'*80)
        return True
    elif type == "half":
        if epoch >= total_epoch / 2:
            print('#'*80)
            print('Push Forward Epoch')
            print('#'*80)
            return True
        else:
            return False
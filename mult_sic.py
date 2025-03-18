import os
import random
import torch
import h5py
from mpi4py import MPI
import torch.distributed as dist
from tqdm import tqdm  # Added tqdm import
from src.models.dqfno import DQFNO
from scripts.utils import create_run_directory, initialize_model, get_loss_object, plot_and_save_loss
from src.losses.custom_losses import MultiTaskLoss
from src.losses.data_losses import LpLoss, H1Loss
from src.data.data_utils import get_data_loader, get_test_loader, PushForwardDataSet, push_forward

# Configuration
path = "/work/10407/anthony50102/frontera/data/hw2d_sim/t600_d256x256_raw"

# Initialize MPI and get rank info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

# Set GPU device for each rank
if torch.cuda.is_available():
    local_gpu = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_gpu)
    device = torch.device("cuda", local_gpu)
else:
    device = torch.device("cpu")

config = {
    'data_dir': path,
    'modes': [[16, 16], [32, 32], [8, 8]],
    'in_channels': 1,
    'out_channels': 1,
    'hidden_channels': 16,
    'n_layers': 4,
    'dx': 0.1636246173744684,
    'derived_type': 'direct',
    'device': device,
    'lr': 0.001,
    'weight_decay': 0.01,
    'losses': ['lp', 'h1', 'derived'],
    'loss_weights': [0.4, 0.4, 0.2],
    'n_epochs': 10,
    'chunk_size': 200,
    'max_pf_step': 8
}

def get_list_files(dir, rand=True):
    files = os.listdir(dir)
    if rand:
        random.shuffle(files)
    return files

def get_pf_steps(curr_epoch, total_epoch, max_pf_step):
    initial_phase = 3
    final_phase = total_epoch - 2
    if curr_epoch < initial_phase:
        return 1
    elif curr_epoch >= final_phase:
        return random.randint(1, max_pf_step)
    else:
        return 1 + int((max_pf_step - 1) * (curr_epoch - initial_phase) / (final_phase - initial_phase))

def get_chunk(file_path, chunk_size, chunk_index):
    with h5py.File(file_path, 'r') as f:
        start = chunk_index * chunk_size
        end = start + chunk_size
        if start >= f['gamma_n'].shape[0]:
            raise IndexError("Chunk index is out of range.")

        n = torch.tensor(f['density'][start:end], dtype=torch.float32)
        e = torch.tensor(f['omega'][start:end], dtype=torch.float32)
        p = torch.tensor(f['phi'][start:end], dtype=torch.float32)
        gn = torch.tensor(f['gamma_n'][start:end], dtype=torch.float32).unsqueeze(0)
        
        state = torch.stack((n, e, p)).permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)
    return (state.to(device, non_blocking=True), gn.to(device, non_blocking=True))

if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(world_size)
    if "MASTER_ADDR" not in os.environ:
        # Set this to the IP or hostname of the master node. "localhost" may work in single-node testing.
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    # Initialize the process group using environment variables set by the launcher
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
    torch.cuda.empty_cache()
    
    # Initialize model and move it to the appropriate device
    model = DQFNO(
        modes=config['modes'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        hidden_channels=config['hidden_channels'],
        n_layers=config['n_layers'],
        dx=config['dx'],
        derived_type=config['derived_type'],
    ).to(device)
    
    # Wrap the model with DistributedDataParallel (DDP)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device.index] if torch.cuda.is_available() else None)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    selector_state = lambda y_pred, y: (y_pred[0], y[0])
    selector_derived = lambda y_pred, y: (y_pred[1], y[1])

    losses = []
    selectors = []
    for loss, weight in zip(config['losses'], config['loss_weights']):
        if loss == 'lp':
            losses.append(LpLoss(d=4, p=2, reduction='mean'))
            selectors.append(selector_state)
        elif loss == 'h1':
            losses.append(H1Loss(d=2))
            selectors.append(selector_state)
        elif loss == 'derived':
            losses.append(torch.nn.MSELoss())
            selectors.append(selector_derived)

    loss_obj = MultiTaskLoss(
        loss_functions=losses,
        scales=config['loss_weights'],
        multi_output=True,
        input_selectors=selectors
    )
    
    if rank == 0:
        print(f"Training using device: {device} on rank {rank} with world size {world_size}")
    
    # Training loop
    for epoch in tqdm(range(config['n_epochs']), desc=f"Epochs Rank {rank}"):
        running_loss = 0.0
        # Get list of files and partition them among ranks
        file_list = get_list_files(config['data_dir'])
        local_files = file_list[rank::world_size]
        
        for file in tqdm(local_files, desc=f"Files Rank {rank}", leave=False):
            pf_steps = get_pf_steps(epoch, config['n_epochs'], config['max_pf_step'])
            optimizer.zero_grad()
            
            file_path = os.path.join(config['data_dir'], file)
            input_data = get_chunk(file_path, config['chunk_size'], chunk_index=0)
            
            for _ in range(pf_steps):
                input_data = model(input_data)
            
            target_chunk_index = pf_steps
            target_data = get_chunk(file_path, config['chunk_size'], chunk_index=target_chunk_index)
            
            loss = loss_obj(input_data, target_data)
            del target_data
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
        
        # Aggregate losses across all nodes and compute average loss
        loss_tensor = torch.tensor(running_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
        if rank == 0:
            print(f"Epoch {epoch+1}/{config['n_epochs']} Average Loss: {avg_loss}")

    # Save the model only from rank 0
    if rank == 0:
        # With DDP, the underlying model is in model.module
        model.module.save(".", "model.pth")
    
    dist.destroy_process_group()

import os
import random
import torch
import h5py
from datetime import datetime  # For timestamping
from tqdm import tqdm
from src.models.dqfno import DQFNO
from scripts.utils import create_run_directory, initialize_model, get_loss_object, plot_and_save_loss
from src.losses.custom_losses import MultiTaskLoss
from src.losses.data_losses import LpLoss, H1Loss
from src.data.data_utils import get_data_loader, get_test_loader, PushForwardDataSet, push_forward

# Configuration
path = "/work/10407/anthony50102/frontera/data/hw2d_sim/t600_d256x256_raw"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'data_dir': path,
    'modes': [[16, 16], [32, 32], [8, 8]],
    'in_channels': 1,
    'out_channels': 1,
    'hidden_channels': 8,
    'n_layers': 2,
    'dx': 0.1636246173744684,
    'derived_type': 'direct',
    'device': device,
    'lr': 0.001,
    'weight_decay': 0.01,
    'losses': ['lp', 'h1', 'derived'],
    'loss_weights': [0.4, 0.4, 0.2],
    'n_epochs': 15,
    'chunk_size': 200,
    'max_pf_step': 5
}

def get_list_files(dir, rand=True):
    files = os.listdir(dir)
    if rand:
        random.shuffle(files)
    return files

def get_pf_steps(curr_epoch, total_epoch, max_pf_step):
    initial_phase = 4
    final_phase = total_epoch - 2
    if curr_epoch < initial_phase:
        return 1
    elif curr_epoch >= final_phase:
        return random.randint(1, max_pf_step)
    else:
        return 1 + int((max_pf_step - 1) * (curr_epoch - initial_phase) / (final_phase - initial_phase))

def get_chunk(f, chunk_size, chunk_index):
    start = chunk_index * chunk_size
    end = start + chunk_size
    if start >= f['gamma_n'].shape[0]:
        raise IndexError("Chunk index is out of range.")

    # Use torch.from_numpy to avoid unnecessary data copies
    n = torch.from_numpy(f['density'][start:end]).float()
    e = torch.from_numpy(f['omega'][start:end]).float()
    p = torch.from_numpy(f['phi'][start:end]).float()
    gn = torch.from_numpy(f['gamma_n'][start:end]).float().unsqueeze(0)
    
    state = torch.stack((n, e, p)).permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)
    return (state.to(device, non_blocking=True), gn.to(device, non_blocking=True))

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Initialize model
    model = DQFNO(
        modes=config['modes'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        hidden_channels=config['hidden_channels'],
        n_layers=config['n_layers'],
        dx=config['dx'],
        derived_type=config['derived_type'],
    ).to(device)

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

    print(f"Training using device: {device}")
    for epoch in tqdm(range(config['n_epochs']), desc="Epochs"):
        running_loss = 0.0
        file_list = get_list_files(config['data_dir'])
        for file in tqdm(file_list, desc="Files", leave=False):
            pf_steps = get_pf_steps(epoch, config['n_epochs'], config['max_pf_step'])
            optimizer.zero_grad()
            
            file_path = os.path.join(config['data_dir'], file)
            # Open the HDF5 file only once per file iteration
            with h5py.File(file_path, 'r') as f:
                # Get initial input chunk
                input_data = get_chunk(f, config['chunk_size'], chunk_index=0)
                
                for _ in range(pf_steps):
                    input_data = model(input_data)
                
                # Get the target chunk from the same file handle
                target_data = get_chunk(f, config['chunk_size'], chunk_index=pf_steps)
                loss = loss_obj(input_data, target_data)
            
            loss.backward()
            optimizer.step()
            
            running_loss += float(loss)
        
        print(f"Epoch {epoch+1}/{config['n_epochs']} Loss: {running_loss}")

    # Add a datetime stamp to the model save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(".", f"model_{timestamp}.pth")

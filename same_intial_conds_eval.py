import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.models.dqfno import DQFNO
from same_intial_conds import get_chunk, get_list_files

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model...")
model = DQFNO.load(".", "model.pth", device=device)
model.eval()
print("Model loaded successfully.")

# Configuration
data_dir = "/work/10407/anthony50102/frontera/data/hw2d_sim/t600_d256x256_raw"
chunk_size = 200
max_pf_step = 5
num_plots = 2  # Number of files to visualize

# Select test files (deterministic selection)
print("\nSelecting test files...")
test_files = get_list_files(data_dir, rand=False)[:num_plots]
print(f"Selected test files: {test_files}")

for file in test_files:
    file_path = os.path.join(data_dir, file)
    print(f"\nProcessing file: {file}")

    # Load initial chunk (retrieve both state and gamma_n)
    input_state, input_gamma_n = get_chunk(file_path, chunk_size, chunk_index=0)
    
    # Autoregressive rollout: update both state and gamma_n
    pred_gamma_ns = [input_gamma_n.detach().cpu().numpy()]
    print("\nRunning autoregressive rollout...")
    for step in tqdm(range(max_pf_step + 2), desc="Rollout Progress", leave=True):
        with torch.no_grad():
            input_state, input_gamma_n = model((input_state, input_gamma_n))
        pred_gamma_ns.append(input_gamma_n.detach().cpu().numpy())
    pred_gamma_ns = np.array(pred_gamma_ns).squeeze()

    # Load ground truth for gamma_n for comparison
    true_gamma_ns = []
    print("\nLoading ground truth data...")
    for i in tqdm(range(len(pred_gamma_ns)), desc="Ground Truth Loading", leave=True):
        try:
            _, gt_gamma_n = get_chunk(file_path, chunk_size, chunk_index=i)
            true_gamma_ns.append(gt_gamma_n.detach().cpu().numpy())
        except IndexError:
            print(f"Index {i} out of bounds for ground truth data. Stopping.")
            break
    true_gamma_ns = np.array(true_gamma_ns).squeeze()

    # Plot gamma_n evolution for this file
    plt.figure(figsize=(8, 5))
    plt.plot(true_gamma_ns, label="Ground Truth γₙ", marker="o")
    plt.plot(pred_gamma_ns, label="Predicted γₙ", marker="x")
    plt.xlabel("Time Step")
    plt.ylabel("γₙ")
    plt.title("Gammaₙ Evolution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"predicted_gamma_n_{file}.png")
    plt.close()
    print(f"Saved gammaₙ evolution plot as predicted_gamma_n_{file}.png")

print("\nTesting complete. All plots saved.")

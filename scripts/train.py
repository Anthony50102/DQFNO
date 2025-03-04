import sys
import os
from typing import Callable, List, Tuple
from tqdm import tqdm
import torch
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from src.data.data_utils import get_data_loader, get_test_loader
from src.models.dqfno import DQFNO
from src.losses.custom_losses import MultiTaskLoss
from src.losses.data_losses import LpLoss, H1Loss
 
from utils import create_run_directory

def main() -> None:
    # Read the configuration
    config_name: str = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig("./default_config.yaml", config_name="default", config_folder="config"),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder="../config"),
        ]
    )
    config = pipe.read_conf()

    # Check device
    device = torch.device(config.device)
    if torch.cuda.is_available() and (config.device == "cpu"):
        print(f"Selected CPU but CUDA is availible...")
    if torch.cuda.is_available() and (config.device == "cuda"):
        print(f"Using CUDA...")

    # Create train and test loaders
    train_loader, train_dataset = get_data_loader(
        config.data.train_input_dir,
        config.data.train_target_dir,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        device=device,
    )

    test_loader, test_dataset = get_test_loader(
        config.data.test_input_dir,
        config.data.test_target_dir,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        device=device,
    )

    # Access dx directly
    dx = train_dataset.dx
    if dx != test_dataset.dx:
        raise ValueError("Test and train datasets have differtent dx")

    # Initialize the model
    model = DQFNO(
        modes=config.dqfno.modes,
        in_channels=config.dqfno.data_channels,
        out_channels=config.dqfno.data_channels,
        hidden_channels=config.dqfno.hidden_channels,
        n_layers=config.dqfno.n_layers,
        dx = dx,
        derived_type=config.dqfno.derived_type,
    )

    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )

    # Selectors for different outputs
    selector_state = lambda y_pred, y: (y_pred[0], y[0])
    selector_derived = lambda y_pred, y: (y_pred[1], y[1])

    # Define loss functions
    losses: List[torch.nn.Module] = []
    selectors: List[Callable] = []
    for loss, weight in zip(config.losses.losses, config.losses.weights):
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
        scales=config.losses.weights,
        multi_output=True,
        input_selectors=selectors
    )

    #loss_obj.to(device)

    # Training loop
    for epoch in range(config.opt.n_epochs):
        running_loss = 0.0
        last_loss = 0.0

        for i, (inputs, targets) in tqdm(enumerate(train_loader),
                                         desc=f"Training Epoch #{epoch}",
                                         total=len(train_loader)):
            
            inputs = (inputs[0].to(device), inputs[1].to(device))
            targets = (targets[0].to(device),targets[1].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_obj(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss)
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # Loss per batch
                print(f'  Batch {i + 1} Loss: {last_loss}')
        print(f"Epoch #{epoch}: Loss: {last_loss}")

    # Save and reload model
    run_dir = create_run_directory(config=config)
    model.save(run_dir, "model.pth")
    model = model.load(run_dir, "model.pth")

    # Evaluate model on test set
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = (inputs[0].to(device), inputs[1].to(device))
            targets = (targets[0].to(device),targets[1].to(device))
            outputs = model(inputs)
            loss_fn = torch.nn.L1Loss()
            loss = loss_fn(outputs[0], targets[0])
            print(f"Test Loss: {loss.item()}")
            break  # Only test on the first batch

    print("Finished training")

if __name__ == '__main__':
    main()

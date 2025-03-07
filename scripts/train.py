import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from src.data.data_utils import get_data_loader, get_test_loader
from src.models.dqfno import DQFNO
from src.losses.custom_losses import MultiTaskLoss
from src.losses.data_losses import LpLoss, H1Loss
from utils import create_run_directory, initialize_model, get_loss_object, plot_and_save_loss

def main() -> None:
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/default_config.yaml"
    print(config_file)
    pipe = ConfigPipeline([
        YamlConfig(config_file, config_name="default"),
        # TODO - Fix?
        # ArgparseConfig(infer_types=True, config_name=None, config_file=None),
    ])
    config = pipe.read_conf()

    device = torch.device(config.device)
    if torch.cuda.is_available() and config.device == "cpu":
        print("Selected CPU but CUDA is available...")
    if torch.cuda.is_available() and config.device == "cuda":
        print("Using CUDA...")

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

    dx = train_dataset.dx
    if dx != test_dataset.dx:
        raise ValueError("Test and train datasets have different dx")

    model = initialize_model(config, dx, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.opt.learning_rate, weight_decay=config.opt.weight_decay)
    loss_obj = get_loss_object(config)

    train_losses, test_losses = [], []
    for epoch in range(config.opt.n_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in tqdm(enumerate(train_loader), desc=f"Training Epoch #{epoch}", total=len(train_loader)):
            inputs, targets = (inputs[0].to(device), inputs[1].to(device)), (targets[0].to(device), targets[1].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_obj(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"Epoch #{epoch}: Train Loss: {epoch_train_loss}")

        model.eval()
        test_loss = sum(float(loss_obj(model((inputs[0].to(device), inputs[1].to(device))), (targets[0].to(device), targets[1].to(device)))) for inputs, targets in test_loader) / len(test_loader)
        test_losses.append(test_loss)
        print(f"Epoch #{epoch}: Test Loss: {test_loss}")
        model.train()

    run_dir = create_run_directory(config=config)
    model.save(run_dir, "model.pth")
    plot_and_save_loss(train_losses, test_losses, config.opt.n_epochs, run_dir)
    print("Finished training. Loss plot saved.")

if __name__ == '__main__':
    main()
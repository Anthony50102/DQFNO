import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from src.data.data_utils import get_data_loader, get_test_loader, PushForwardDataSet, push_forward
from src.models.dqfno import DQFNO
from src.losses.custom_losses import MultiTaskLoss
from src.losses.data_losses import LpLoss, H1Loss
from utils import create_run_directory, initialize_model, get_loss_object, plot_and_save_loss

def main() -> None:
    custom_config = False
    if len(sys.argv) < 1:
        raise ValueError("No default config passed")
    else:
        print(f"Using config: {sys.argv[1]}")
        config_file = sys.argv[1]
        sys.argv = sys.argv[1:]
    pipe = ConfigPipeline([
        YamlConfig(config_file, config_name="default"),
        ArgparseConfig(config_name=None, config_file=None),
        YamlConfig()
    ])
    config = pipe.read_conf()

    device = torch.device(config.device)
    if torch.cuda.is_available() and config.device == "cpu":
        print("Selected CPU but CUDA is available...")
    elif torch.cuda.is_available() and config.device == "cuda":
        print("Using CUDA...")
    elif config.device == 'cpu':
        print("Using CPU...")

    train_loader, train_dataset = get_data_loader(
        config.data.train_input_dir,
        config.data.train_target_dir,
        batch_size=config.data.batch_size,
        shuffle=True,
        # Todo - Allow for many workers?
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
    chunk_size = train_dataset.chunk_size
    if dx != test_dataset.dx:
        raise ValueError("Test and train datasets have different dx")

    pf = False
    if config.data.push_forward_dir != None:
        pf_dataclass =PushForwardDataSet(config.data.push_forward_dir, 
                                         config.data.push_forward_steps,
                                         chunk_size=chunk_size)
        # Flag to indicate wether to push foward train
        pf = True


    model = initialize_model(config, dx, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.opt.learning_rate, weight_decay=config.opt.weight_decay)
    loss_obj = get_loss_object(config)

    train_losses, test_losses = [], []
    for epoch in range(config.opt.n_epochs):
        running_loss = 0.0
        # Push Forward Training
        if push_forward(total_epoch=config.opt.n_epochs, 
                        epoch=epoch, 
                        type=config.opt.push_forward_type) and pf:
            # Loop through all files
            for file in pf_dataclass.files:
                optimizer.zero_grad()
                chunk_indices = pf_dataclass.get_chunk(file, rand_pf_step=True)
                input_data = pf_dataclass.get_data(file, chunk_indices[0].item()) 
                for i in range(config.data.push_forward_steps):
                    input_data = model(input_data) 
                
                target_data = pf_dataclass.get_data(file, chunk_indices[-1].item())
                loss = loss_obj(input_data, target_data)
                loss.backward()
                optimizer.step()
                running_loss += float(loss)

            epoch_train_loss = running_loss / len(pf_dataclass.files)
            train_losses.append(epoch_train_loss)
            print(f"Epoch #{epoch}: Train Loss: {epoch_train_loss}")

            # TODO - Implement Test loss
            test_losses.append(epoch_train_loss)

        else:
            # Standard training
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

import sys
from tqdm import tqdm
import os
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from src.data.data_utils import get_data_loader, get_test_loader
from src.models.dqfno import DQFNO
from src.losses.custom_losses import MultiTaskLoss
from src.losses.data_losses import LpLoss, H1Loss

def main():
    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig("./default_config.yaml", config_name="default", config_folder="config"),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder="../config"),
        ]
    )
    config = pipe.read_conf()

    # Create the train and test loaders (you might want to adjust num_workers if needed)
    train_loader, test_loader = get_data_loader(config.data.train_input_dir,
                                   config.data.train_target_dir,
                                   batch_size=config.data.batch_size,
                                   shuffle=True, num_workers=0), \
                                get_test_loader(config.data.train_input_dir,
                                   config.data.train_target_dir,
                                   batch_size=config.data.batch_size,
                                   shuffle=True, num_workers=0)
    
    # TODO - Implement
    # model = get_model(config)
    model = DQFNO(
        modes = config.dqfno.modes,
        in_channels = config.dqfno.data_channels,
        out_channels= config.dqfno.data_channels,
        hidden_channels = config.dqfno.hidden_channels,
        n_layers = config.dqfno.n_layers,
         
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )
    selector_state = lambda y_pred, y: (y_pred[0], y[0])
    selector_derived = lambda y_pred, y: (y_pred[1], y[1])

    losses = []
    selectors = []
    for loss, weight in zip(config.losses.losses, config.losses.weights):
        if loss == 'lp':
            lpLoss = LpLoss(d=4, p=2 , reduction='mean')
            losses.append(lpLoss)
            selectors.append(selector_state)
        if loss == 'h1':
            h1Loss = H1Loss(d=2)
            losses.append(h1Loss)
            selectors.append(selector_state)
        if loss == 'derived':
            derived_loss = torch.nn.MSELoss()
            losses.append(derived_loss)
            selectors.append(selector_derived)

    loss_obj = MultiTaskLoss(
        loss_functions=losses,
        scales=config.losses.weights,
        multi_output=True,
        input_selectors=selectors
    ) 

    for epoch in range(config.opt.n_epochs):
        running_loss = 0.
        last_loss = 0.
        for i, data in tqdm(enumerate(train_loader),
                            desc=f"Training Epoch #{epoch}",
                            total=len(train_loader)
                            ):
            inputs, targets = data # Each input is a list of [State (B,C,V,T,X,Y), Derived (B,T,D)]

            optimizer.zero_grad()

            outputs = model(inputs)
            # TODO - Fix this to allow for actual different batch sizes
            loss = loss_obj(outputs, targets)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += float(loss)
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
        print(f"Epochs #{epoch}: Loss: {last_loss}")
    
    model.save("../", "model.pth")
    model = model.load("../", "model.pth")
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            print(type(inputs))
            outputs = model(inputs)
            lossl = torch.nn.L1Loss()
            loss = lossl(outputs[0], targets[0])
            print(loss)
            break
    print("Finished training")




if __name__ == '__main__':
    main()

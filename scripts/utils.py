import os
import json
from datetime import datetime

def create_run_directory(config, base_dir='./runs'):
    """
    Creates a unique directory for the current training run and saves meta data.

    Parameters:
        config (dict or object): The training configuration containing hyperparameters 
                                 and other relevant settings. If not a dict, an attempt 
                                 is made to convert it via __dict__.
        base_dir (str): The base folder where run directories will be created (default: './runs').

    Returns:
        str: The path to the newly created run directory.
    """
    # Ensure the base directory exists.
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a unique run ID based on the current timestamp.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=False)
    
    # Convert config to a dictionary if it isn't already.
    if isinstance(config, dict):
        config_to_save = config
    else:
        try:
            config_to_save = config.__dict__
        except Exception:
            config_to_save = str(config)
    
    # Save the configuration as a JSON file.
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=4, default=str)
    
    # Save additional meta data (e.g., run start time, run ID).
    meta_data = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config_file": config_path,
    }
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    
    return run_dir

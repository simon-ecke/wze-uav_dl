import torch
import os
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

def create_writer(target_dir: str,
                  experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    Where timestamp is the current date in YYYY-MM format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(target_dir=logs,
                               experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="logs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m") # returns current date in YYYY-MM format

    if extra:
        # Create log directory path
        log_dir = os.path.join(target_dir, timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join(target_dir, timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

import os
import argparse
import pathlib
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, utils as vutils
import trimesh as tm
from pytorchtrainer.trainer import Trainer
from notiontoolkit.tqdm_notion import tqdm_notion
from dotenv import load_dotenv
from torchinfo import summary

from network import DefTet
from voxelizedMeshDataset import VoxelizedMeshDataset
from meshFileLoader import load_mesh

########################
# Constants
########################
RESULTS_DIRECTORY = "results"
NOTION_NAME = None
########################

########################
# Hyperparameters
########################
epochs = 100
learning_rate = 3e-4


########################

########################
# Model Definition
########################
def create_model(batch_size):
    in_size = 32
    latent_size = 8
    in_channels = 1

    # Provide unit cube for deformation.
    template_mesh = load_mesh("./unit_cube.vtk")

    model = DefTet(
        in_size=in_size,
        latent_size=latent_size,
        in_channels=in_channels,
        template_mesh=template_mesh
    )

    # Dummy forward pass to initialise weights before distributing.
    model(torch.zeros(batch_size, in_channels, in_size, in_size, in_size))

    return model
########################

########################
# Dataset Setup
########################
def create_data_loaders(rank: int, world_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    training_data = VoxelizedMeshDataset(os.path.join(os.getenv("DATA_PATH"), "train"))
    validation_data = VoxelizedMeshDataset(os.path.join(os.getenv("DATA_PATH"), "test"))


    # Create a sampler for distributed loading.
    sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True, seed=42)

    # Make Dataloaders for each dataset.
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False, num_workers=16, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader
########################

def main(rank: int, epochs: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    # Put model onto the relevant device for this process rank.
    device = torch.device(f'cuda:{rank}')
    print(f'Rank {rank} using {torch.cuda.get_device_name(device)}.')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

########################
# Train Model
########################
    # Define optimiser and loss functions.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    # Override fwd method in Trainer.
    class DefTetTrainer(Trainer):
        def step(self, batch):
            pass
            
    # Instantiate the trainer.
    trainer = DefTetTrainer(model, train_loader, val_loader, optimizer, device=device, ddp=True, rank=rank)
    
    # Enable Notion integration to track training progress.
    if NOTION_NAME is not None:
        print("Notion integration enabled.")
        trainer.tqdm = tqdm_notion
        trainer.tqdm_kwargs = {"page_title": NOTION_NAME}

    # Define a callback to print the loss and accuracy each epoch.
    def plot():
        trainer.plot_loss(save_path=RESULTS_DIRECTORY, quiet=True, yscale='log')
        trainer.plot_accuracy(save_path=RESULTS_DIRECTORY, quiet=True)
        trainer.save_model(os.path.join(RESULTS_DIRECTORY, "classifier.pth"))

    trainer.set_callback(plot, np.linspace(1, epochs, epochs//5,  endpoint=False, dtype=int))

    # Train the model. Catch keyboard interrupts for clean exiting.
    try:
        trainer.train(epochs, quiet=args.quiet)
    except KeyboardInterrupt as e:
        print("User interrupted training, saving current results and terminating.")
        return trainer
    
    return trainer
########################


if __name__ == '__main__':
########################
# Args and Env Vars
########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', default=False, action="store_true", help="Surpress script output for headless environments. Default=False.")
    parser.add_argument('--notion', type=str, help='Name for Notion entry. Will not use Notion if not set.')
    parser.add_argument('--dataset', type=str, default='SynthShape', help="Which dataset to use. Default=cifar10.")
    parser.add_argument('--batchsize', type=int, default=32, help="Batch size. Default=128")
    args = parser.parse_args()
    load_dotenv()
########################
# Set conditional constants.
    RESULTS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, args.dataset)
    pathlib.Path(RESULTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    NOTION_NAME = args.notion

    # Setup DDP things.
    rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv('WORLD_SIZE'))

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')

    # Setup data and train the model.
    train_loader, val_loader = create_data_loaders(rank, world_size, args.batchsize)

    trainer = main(rank=rank,
                 epochs=epochs,
                 model=create_model(args.batchsize),
                 train_loader=train_loader,
                 val_loader=val_loader)

########################
# Save Trainer
########################
    if rank == 0:
        trainer.plot_loss(save_path=RESULTS_DIRECTORY, quiet=args.quiet, yscale='log')
        trainer.plot_accuracy(save_path=RESULTS_DIRECTORY, quiet=args.quiet)
        trainer.save_model(os.path.join(RESULTS_DIRECTORY, "model.pth"))
########################
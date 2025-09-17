import argparse
import torch
import datetime
import yaml
import os
import time

from sysid_models import *
from datasets import *
from utils import train, evaluate

parser = argparse.ArgumentParser(description="DiffSYsId")
parser.add_argument('--device', default='cuda:0', help='Device')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
# Set the model and dataset
parser.add_argument("--model", type=str, default="LotkaVolterra")
parser.add_argument("--dataset", type=str, default="data/LotkaVolterra_S_10000_T_20.0_dt_0.001_seq_len_60_seed_0/dataset.pt")
parser.add_argument("--config", type=str, default="LotkaVolterra.yaml")


args = parser.parse_args()
print(args)

# Load Data
dataset = torch.load(args.dataset, weights_only=False)
# normalize time series values and parameters
dataset = NormalizedDataset(dataset)

# training the model
if args.modelfolder == "":

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(yaml.dump(config, indent=4, sort_keys=False))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/" + args.model + "_fold" + str(args.nfold) + "_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.yaml", "w") as f:
        yaml.dump(config, f, indent=4, sort_keys=False)

    # get dataloaders
    train_loader , valid_loader, test_loader = get_dataloader(
        dataset,
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"])

    # specify the model
    model = DiffSysId(config, args.device).to(args.device)

    # train the model
    start_time = time.time()
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")
else:
    path = "./save/" + args.modelfolder + "/config.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(yaml.dump(config, indent=4, sort_keys=False))

    # get dataloaders
    train_loader , valid_loader, test_loader = get_dataloader(
        dataset,
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"])

    model = DiffSysId(config, args.device).to(args.device)
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

model.eval()
batch = next(iter(test_loader))
# Evaluate the model
identified_parameters, _, _ = model.evaluate(batch, n_samples=args.nsample) # (B, nsample, param_dim)
mean = dataset.parameters_mean.to(args.device)
std = dataset.parameters_std.to(args.device)
identified_parameters = identified_parameters * std + mean
parameters = batch["parameters"].to(args.device)
parameters = parameters * std + mean
print("Identified Parameters: ", identified_parameters[0])
print("Parameters: ", parameters[0])


# evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername="./save/" + args.modelfolder + "/")
# Note: The evaluate function in utils.py is designed for time series imputation, not system identification
# The main evaluation results (identified parameters vs true parameters) are already displayed above






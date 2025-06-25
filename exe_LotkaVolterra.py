import argparse
import torch
import datetime
import json
import yaml
import os

from sysid_models import DiffSysId_LV
from dataset_LotkaVolterra import get_dataloader_LotkaVolterra, NormalizedDataset
from utils import train, evaluate

parser = argparse.ArgumentParser(description="DiffSYsId")
parser.add_argument("--config", type=str, default="diffsysid.yaml")
parser.add_argument('--device', default='cuda:0', help='Device')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--dataset", type=str, default="data/LotkaVolterra_S_10000_T_20.0_dt_0.01_seed_0/dataset.pt")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/LV_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# Load Data
dataset = torch.load(args.dataset, weights_only=False)
# normalize time series values and parameters
dataset = NormalizedDataset(dataset)

# get dataloaders
train_loader , valid_loader, test_loader = get_dataloader_LotkaVolterra(
    dataset,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"])

model = DiffSysId_LV(config, args.device, config["model"]["target_dim"]).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
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

# evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)




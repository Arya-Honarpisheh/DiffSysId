This is the Read Me File.

## datasets.py

`get_dataloader()` - Creates train/validation/test data splits (70%/10%/20%) with 5-fold cross-validation support for reproducible experiments.

`NormalizedDataset` - Wrapper class that normalizes dataset features to zero mean and unit variance for improved neural network training.

`LVModel()` - Simulates the Lotka-Volterra predator-prey differential equations using numerical integration.

`LotkaVolterraDataset` - PyTorch dataset class that generates synthetic predator-prey time series data with random parameters, initial conditions, and realistic noise.

`LorenzModel()` - Simulates the Lorenz attractor differential equations using numerical integration.

`LorenzDataset` - PyTorch dataset class that generates synthetic chaotic Lorenz system time series data with random parameters, initial conditions, and realistic noise.

This is the Read Me File.

## datasets.py

`get_dataloader()` - Creates train/validation/test data splits (70%/10%/20%) with 5-fold cross-validation support for reproducible experiments.

`NormalizedDataset` - Wrapper class that normalizes dataset features to zero mean and unit variance for improved neural network training.

`LVModel()` - Simulates the Lotka-Volterra predator-prey differential equations using numerical integration.

`LotkaVolterraDataset` - PyTorch dataset class that generates synthetic predator-prey time series data with random parameters, initial conditions, and realistic noise.

`LorenzModel()` - Simulates the Lorenz attractor differential equations using numerical integration.

`LorenzDataset` - PyTorch dataset class that generates synthetic chaotic Lorenz system time series data with random parameters, initial conditions, and realistic noise.

## Dataset Creation Scripts

`create_LotkaVolterra.py` - Generates synthetic Lotka-Volterra predator-prey datasets with configurable parameters.

`create_Lorenz.py` - Generates synthetic Lorenz attractor datasets with configurable parameters.

## sysid_models.py

`DiffSysId_base` - Base class implementing diffusion-based system identification using denoising diffusion probabilistic models (DDPM) for parameter estimation from time series data. **Input**: Time series observations (B, K, L), time points (B, L), and true parameters (B, P). **Output**: Predicted noise in the forward path in training and identified parameters (B, n_samples, P) during evaluation. During training, random diffusion steps are selected to add noise to parameters and output the error between real and predicted noise. During inference, the reverse process iterates from num_steps-1 to 0, gradually denoising parameters to identify the underlying system parameters.

`DiffSysId` - Concrete implementation for K dimensional systems (e.g., Lotka-Volterra) that processes batch data and formats inputs for the diffusion model's forward pass.

## noise_predictor.py

`noise_predictor` - Deep Architecture that predicts the noise added to parameters during the diffusion forward process for system identification. **Input**: Noisy parameters (B, P), time series observations (B, K, L), time points (B, L), and diffusion step (B). **Output**: Predicted noise (B, P). The model combines three specialized embedding modules: (1) `DiffusionEmbedding` - Creates sinusoidal embeddings for diffusion timesteps using frequency-based encoding, (2) `ParameterEmbedding` - Multi-layer perceptron that embeds noisy parameter vectors, and (3) `TimeSeriesEmbedding` - Transformer-based encoder that processes time series data with positional encoding, projecting time series through hidden layers before applying multi-head self-attention and global pooling. All embeddings are concatenated and passed through `OutputProjection` to predict the noise that should be removed from parameters during the reverse diffusion process.


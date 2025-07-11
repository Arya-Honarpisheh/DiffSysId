import numpy as np
import torch
import torch.nn as nn
from noise_predictor import noise_predictor

class DiffSysId_base(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        """
        This Module calculates the l2 norm error between the predicted and true noise added
        in the diffusion proccess. In training mode it calculates the error for some random
        time step in the diffusion process, while in validation mode it calculates the error
        as the average over all time steps.
        B: batch size
        K: time series dim
        L: time series length
        P: number of parameters
        """

        self.device = device
        self.ts_dim = config["model"]["ts_dim"]  # K
        self.param_dim = config["model"]["param_dim"] # P

        config_diff = config["diffusion"]
        # self.emb_time_dim = config_diff["time_embedding_dim"] # embedding dimension for time used in positional encoding
        
        self.diffmodel = noise_predictor(config_diff, self.param_dim, self.ts_dim, self.device)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta # (num_steps)
        self.alpha = np.cumprod(self.alpha_hat) # (num_steps)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1) # (num_steps, 1)

    def calc_loss_valid(
        self, observed_data, parameters, observed_tp, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, parameters, observed_tp, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, parameters, observed_tp, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device) # (B)
        else:
            # for training, we do not need to pass the argument set_t.
            t = torch.randint(0, self.num_steps, [B]).to(self.device) # (B)
        # note that t is a tensor of dimension (B). Thus, we can use advanced indexing to get
        # alpha for each sample in the batch.
        current_alpha = self.alpha_torch[t]  # (B, 1)

        noise = torch.randn(B, self.param_dim).to(self.device) # (B, P)
        noisy_parameters = (current_alpha ** 0.5) * parameters + (1.0 - current_alpha) ** 0.5 * noise # (B, P)

        predicted = self.diffmodel(noisy_parameters, observed_data, observed_tp, t)  # (B, P)

        # expand noise along the time dimension
        residual = noise - predicted
        loss = (residual ** 2).mean()
        return loss

    def identify(self, observed_data, observed_tp, n_samples):
        B, K, L = observed_data.shape

        generated_parameters = torch.zeros(B, n_samples, self.param_dim).to(self.device) # (B, nsample, P)

        for i in range(n_samples):

            current_parameters = torch.randn(B, self.param_dim).to(self.device) # (B, P)

            for t in range(self.num_steps - 1, -1, -1):

                diffusion_step = (torch.ones(B) * t).long().to(self.device) # (B)

                predicted = self.diffmodel(current_parameters, observed_data, observed_tp, diffusion_step) # (B, P)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_parameters = coeff1 * (current_parameters - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_parameters)  # (B, P)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_parameters += sigma * noise # (B, P)

            generated_parameters[:, i] = current_parameters.detach() # (B, P)

        return generated_parameters # (B, nsample, P)

    def forward(self, batch, is_train=1):
        (
            observed_tp, # (B, L)
            observed_data, # (B, K, L)
            observed_params # (B, P)
        ) = self.process_data(batch)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, observed_params, observed_tp, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_tp, # (B, L)
            observed_data, # (B, K, L)
            _
        ) = self.process_data(batch)

        with torch.no_grad():

            identified_parameters = self.identify(observed_data, observed_tp, n_samples) # (B, nsample, P)

        return identified_parameters, observed_data, observed_tp
    
class DiffSysId(DiffSysId_base):
    def __init__(self, config, device):
        super().__init__(config, device)

    def process_data(self, batch):
        observed_data = batch["x"].to(self.device).float() # (B, L, 2)
        observed_tp = batch["time"].to(self.device).float() # (B, L)
        observed_params = batch["parameters"].to(self.device).float() # (B, 4)

        observed_data = observed_data.permute(0, 2, 1) # (B, 2, L)

        return (
            observed_tp, # (B, L)
            observed_data, # (B, 2, L)
            observed_params # (B, 4)
        )
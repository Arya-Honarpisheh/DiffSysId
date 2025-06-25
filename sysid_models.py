import numpy as np
import torch
import torch.nn as nn
from diffusion_models import noise_predictor

class DiffSysId_base(nn.Module):
    def __init__(self, ts_dim, config, device):
        super().__init__()
        self.device = device
        self.ts_dim = ts_dim # the number of states

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.ts_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        self.param_dim = config["model"]["param_dim"]
        ### we should fix this later, it must take the noisy parameters, so it takes param_dim as an argument ###
        self.diffmodel = noise_predictor(config_diff, self.param_dim, self.ts_dim)

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

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp):
        
        B, L = observed_tp.shape

        # observed_tp is the time points or positions
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.ts_dim, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.ts_dim, device=self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

    def calc_loss_valid(
        self, observed_data, parameters, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, parameters, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, parameters, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1)

        noise = torch.randn(B, self.param_dim).to(self.device) # (B,param_dim)
        noisy_parameters = (current_alpha ** 0.5) * parameters + (1.0 - current_alpha) ** 0.5 * noise

        predicted = self.diffmodel(noisy_parameters, observed_data, side_info, t)  # (B, param_dim)

        # expand noise along the time dimension
        residual = noise - predicted
        loss = (residual ** 2).mean()
        return loss

    def identify(self, observed_data, side_info, n_samples):
        B, K, L = observed_data.shape

        generated_params = torch.zeros(B, n_samples, self.param_dim).to(self.device) # (B, nsample, param_dim)

        for i in range(n_samples):

            current_params = torch.randn(B, self.param_dim).to(self.device) # (B, param_dim)

            for t in range(self.num_steps - 1, -1, -1):

                predicted = self.diffmodel(current_params, observed_data, side_info, torch.tensor([t]).to(self.device)) # (B, param_dim)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_params = coeff1 * (current_params - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_params)  # (B, param_dim, L)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_params += sigma * noise

            generated_params[:, i] = current_params.detach()
        return generated_params

    def forward(self, batch, is_train=1):
        (
            observed_tp,
            observed_data,
            observed_params
        ) = self.process_data(batch)

        side_info = self.get_side_info(observed_tp)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, observed_params, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_tp,
            observed_data,
            _
        ) = self.process_data(batch)

        with torch.no_grad():

            side_info = self.get_side_info(observed_tp)

            identified_parameters = self.identify(observed_data, side_info, n_samples)

        return identified_parameters, observed_data, observed_tp
    
class DiffSysId_LV(DiffSysId_base):
    def __init__(self, config, device, ts_dim=2):
        super(DiffSysId_LV, self).__init__(ts_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["x"].to(self.device).float()
        observed_tp = batch["time"].to(self.device).float()
        observed_params = batch["parameters"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1) # (B, K, L)

        return (
            observed_tp,
            observed_data,
            observed_params
        )
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)  # (num_steps,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (num_steps,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (num_steps,dim*2) = (num_steps,embedding_dim)
        return table

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step] # (B, embedding_dim)
        x = self.projection1(x) # (B, projection_dim)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x # (B, projection_dim)
    
class ParameterEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, parameters):
        return self.network(parameters)  # (B, output_dim)
    
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, input_proj_hidden_dims, embedding_dim, nheads, nlayers, device):
        super().__init__()

        self.device = device
        self.embedding_dim = embedding_dim

        layers = []
        for hidden_dim in input_proj_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.input_projection = nn.Sequential(*layers)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nheads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)  # (B, L, d_model)
        position = pos.unsqueeze(2) # (B, L, 1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe # (B, L, d_model)
    
    def forward(self, observed_ts, observed_tp):

        x = observed_ts.permute(0, 2, 1)  # (B, L, K)
        x = self.input_projection(x) # (B, L, embedding_dim)
        embeded_time = self.time_embedding(observed_tp, self.embedding_dim)  # (B, L, embedding_dim)
        x = x + embeded_time  # (B, L, embedding_dim)
        x = self.encoder(x) # (B, L, embedding_dim)
        x = x.mean(dim=1).squeeze(1) # (B, embedding_dim)

        return x 

class OutputProjection(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # (B, output_dim)

class noise_predictor(nn.Module):
    def __init__(self, config, param_dim, ts_dim, device):
        super().__init__()
        """
        param_dim: dimension of the parameters to be predicted
        ts_dim: dimension of the time series data
        config: configuration dictionary containing diffusion hyperparameters
        """
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        ) # embeds the diffusion time step

        self.parameter_embedding = ParameterEmbedding(
            input_dim=param_dim,
            output_dim=config["parameter_embedding_dim"],
            hidden_dims=config["parameter_embedding_hidden_dims"]
        )

        self.ts_embedding = TimeSeriesEmbedding(
            input_dim=ts_dim,
            input_proj_hidden_dims=config["ts_embedding_input_proj_hidden_dims"],
            embedding_dim=config["ts_embedding_dim"],
            nheads=config["ts_embedding_nheads"],
            nlayers=config["ts_embedding_nlayers"],
            device=device
        )

        self.output_projection = OutputProjection(
            input_dim=config["diffusion_embedding_dim"] + config["parameter_embedding_dim"] + config["ts_embedding_dim"],
            output_dim=param_dim,
            hidden_dims=config["output_projection_hidden_dims"]
        )

    def forward(self, parameters, observed_data, observed_tp, diffusion_step):

        embeded_diffusion_step = self.diffusion_embedding(diffusion_step)  # (B, diffusion_embedding_dim)

        embeded_parameters = self.parameter_embedding(parameters)  # (B, parameter_embedding_dim)

        embeded_ts = self.ts_embedding(observed_data, observed_tp) # (B, ts_embedding_dim)

        # Concatenate all embeddings
        combined_embedding = torch.cat([embeded_diffusion_step, embeded_parameters, embeded_ts], dim=-1)

        # Pass through output projection
        predicted_noise = self.output_projection(combined_embedding) # (B, P)

        return predicted_noise  # (B, P)

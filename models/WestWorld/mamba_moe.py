import torch
import torch.nn as nn
from mamba_ssm import Mamba

#####################
#  MoE model definitions
#####################

class MambaConfig:
    """
    Configuration class defining the hyperparameters for Mamba MoE.
    """
    def __init__(self, 
                 num_layers=12, 
                 hidden_size=768, 
                 state_size=16, 
                 conv_dimension=4, 
                 expansion_factor=2, 
                 num_experts=4, 
                 top_k=2, 
                 ffn_hidden_size=3072, 
                 layernorm_epsilon=1e-5,
                 use_switch_mlp=False,  # Whether to use SwitchMLP; not used in the current version
                 device="cuda"):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_dimension = conv_dimension
        self.expansion_factor = expansion_factor
        self.num_experts = num_experts
        self.top_k = top_k
        self.ffn_hidden_size = ffn_hidden_size
        self.layernorm_epsilon = layernorm_epsilon
        self.use_switch_mlp = use_switch_mlp  # Choose between MLP and SwitchMLP
        self.device = device  # Device setting

class MLP(nn.Module):
    """
    Standard MLP layer used for MoE experts.
    """
    def __init__(self, config: MambaConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer = layer_idx
        self.linear_fc1 = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=True, device=config.device)
        self.linear_fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=True, device=config.device)
        self.activation_func = nn.GELU()

    def forward(self, hidden_states):
        intermediate = self.activation_func(self.linear_fc1(hidden_states))
        output = self.linear_fc2(intermediate)
        return output

def sinkhorn(cost, tol=0.0001):
    """
    Sinkhorn routing function for MoE with balanced expert assignment.
    """
    cost = torch.exp(2.0 * cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = 1 / (cost.size(1) * torch.sum(cost, 0))
    eps = 1e-8
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)

class SwitchMLP(nn.Module):
    """
    MoE computation layer for the Switch Transformer.
    - Uses Top-1 expert selection
    - Supports Sinkhorn load balancing
    """
    def __init__(self, config: MambaConfig, layer_idx=None):
        super().__init__()
        self.layer = layer_idx
        self.config = config
        self.num_experts = config.num_experts
        self.router = nn.Linear(config.hidden_size, self.num_experts).to(config.device)
        self.routing_mode = "sinkhorn"  # Use sinkhorn by default
        self.route_algo = sinkhorn
        self.router_activation = torch.sigmoid  # Sinkhorn uses sigmoid
        self.local_experts = nn.ModuleList([
            MLP(config, layer_idx=layer_idx) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states):
        hidden_shape = hidden_states.shape  # [batch*seq, hidden_size], flattened below
        route = self.router(hidden_states).view(-1, self.num_experts)
        if self.training:  # Use Sinkhorn for load balancing during training
            with torch.no_grad():
                sinkroute = sinkhorn(route.detach().to(dtype=torch.float32))
                _, max_ind = torch.max(sinkroute, dim=1)
            max_prob = sinkroute[torch.arange(route.size(0)), max_ind].unsqueeze(-1)
        else:  # Use sigmoid + top-1 directly during inference
            route = torch.sigmoid(route)
            max_prob, max_ind = torch.max(route, dim=1)
            max_prob = max_prob.unsqueeze(-1)
        output_total = torch.zeros_like(hidden_states.view(-1, hidden_shape[-1]))
        for i, expert in enumerate(self.local_experts):
            local_indices = (max_ind == i).nonzero().squeeze(-1)
            if local_indices.numel() > 0:
                expert_output = expert(hidden_states.view(-1, hidden_shape[-1])[local_indices])
                output_total[local_indices] = expert_output
        return (output_total * max_prob).view(hidden_shape)
    
class MambaMoELayer(nn.Module):
    """
    Mamba-MoE Layer:
    - Split off the special CLS token for gating.
    - Apply Mamba SSM on the token embeddings.
    - Use the CLS token to compute routing logits for a standard MoE.
    """
    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        from mamba_ssm import Mamba
        self.mamba = Mamba(
            d_model=config.hidden_size,
            d_state=config.state_size,
            d_conv=config.conv_dimension,
            expand=config.expansion_factor,
            layer_idx=layer_idx,
        )
        if config.use_switch_mlp:
            self.moe = SwitchMLP(config, layer_idx=layer_idx)
        else:
            self.moe = nn.ModuleList([MLP(config, layer_idx=layer_idx)
                                      for _ in range(config.num_experts)])
            self.gate = nn.Linear(config.hidden_size, config.num_experts)

    def forward(self, hidden_states):
        """
        hidden_states: [B, T_seq + 1, D], where the last token is the CLS token.
        """
        # 1) Mamba processing
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states + self.mamba(hidden_states)

        # 2) Split tokens vs. CLS
        token_states = hidden_states[:, :-1, :]   # [B, T, D]
        cls_token    = hidden_states[:, -1:, :]   # [B, 1, D]

        # 3) Compute gating scores from CLS token
        cls_vec       = self.norm2(cls_token).squeeze(1)      # [B, D]
        gating_scores = torch.softmax(self.gate(cls_vec), dim=-1)  # [B, E]
        # Prepare for broadcasting: [B, 1, 1, E]
        gating = gating_scores.unsqueeze(1).unsqueeze(1)

        # 4) Expert outputs: stack along a new expert dimension
        #    results in [B, T, D, E]
        experts_out = torch.stack([expert(token_states) for expert in self.moe], dim=-1)
        # Weighted sum across experts -> [B, T, D]
        token_processed = torch.sum(experts_out * gating, dim=-1)

        # 5) Residual skip connection on tokens
        token_output = token_states + token_processed    # [B, T, D]

        # 6) Re-attach CLS token
        out = torch.cat([token_output, cls_token], dim=1)  # [B, T+1, D]
        return out

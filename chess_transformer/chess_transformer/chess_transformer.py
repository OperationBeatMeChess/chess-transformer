import torch
from torch import nn
import timm
import numpy as np
from torch.cuda.amp import GradScaler
from adversarial_gym.chess_env import ChessEnv


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        out = self.linear1(self.norm1(x))
        out = self.linear2(self.gelu(out))
        out = self.norm2(out + x)
        return out


class ChessTransformer(nn.Module):
    """
    Creates an OBM ChessNetwork that outputs a value and action for a given
    state/position. 
    
    The network uses a feature extraction backbone from the Pytorch Image Model library (timm)
    and feeds the ouput of that into two separate prediction heads.

    The output of the policy network is a vector of size action_dim = 4672
    and the ouput of the value network is a single value.

    """

    def __init__(self, device = 'cuda', base_lr = 0.0009, max_lr = 0.009):
        super().__init__()
        
        self.swin_transformer = timm.create_model(
            'swin_large_patch4_window7_224', 
            pretrained=False,
            img_size=8, 
            patch_size=1,
            window_size=2, 
            in_chans=1,
        ).to(device)

        self.action_dim = 4672
        action_embed_dim = 512
        self.action_embed = nn.Embedding(self.action_dim, action_embed_dim).to(device)
        self.device = device

        num_features = self.swin_transformer.head.in_features

        # Policy head
        self.policy_head = nn.Sequential(
            ResidualBlock(num_features, 2*num_features),
            nn.Linear(num_features, 2*num_features),
            nn.GELU(),
            nn.Linear(2*num_features, self.action_dim),
        ).to(device)
        
        # # Value head
        self.value_head = nn.Sequential(
            ResidualBlock(num_features, 2*num_features),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, 1),
            nn.Tanh()
        ).to(device)

        self.critic_head = nn.Sequential(
            ResidualBlock(num_features + action_embed_dim, 2*num_features),
            nn.Linear(num_features + action_embed_dim, num_features),
            nn.GELU(),
            nn.Linear(num_features, 1),
            nn.Tanh()
        ).to(device)
        
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()
        self.critic_loss = nn.MSELoss()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            # Assuming 8x8 array from chess env. Convert to (1,1,8,8) tensor
            x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0).unsqueeze(0)
        
        # Action and value prediction
        features = self.swin_transformer.forward_features(x) # Pooled (B, 1, 1, 1536)
        features = features.view(features.shape[0], -1)
        action_logits = self.policy_head(features)
        board_val = self.value_head(features)
        
        # Critic (S,A) -> R prediction
        action = torch.argmax(action_logits, dim=1)
        action = self.action_embed(action)
        critic_feats = torch.cat((features, action), dim=1)
        critic_val = self.critic_head(critic_feats)

        return action_logits, board_val, critic_val

    def get_action(self, state, legal_moves, sample_n=1):
        """ Randomly sample from top_n legal actions given input state"""

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        features = self.swin_transformer.forward_features(state)
        features = features.view(features.shape[0], -1)
        
        policy_logits = self.policy_head(features)

        legal_actions = [ChessEnv.move_to_action(move) for move in legal_moves]
        return self.to_action(policy_logits, legal_actions, top_n=sample_n)
    
    def to_action(self, action_logits, legal_actions, top_n):
        """ Randomly sample from top_n legal actions given output action logits """

        if len(legal_actions) < top_n: top_n = len(legal_actions)

        action_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        action_probs_np = action_probs.detach().cpu().numpy().flatten()

        # Set non legal-actions to = -inf so they aren't considered
        mask = np.ones(action_probs_np.shape, bool)
        mask[legal_actions] = False
        action_probs_np[mask] = -np.inf

        # sample from indices of the top-n policy probs
        top_n_indices = np.argpartition(action_probs_np, -top_n)[-top_n:]
        action = np.random.choice(top_n_indices)
        
        log_prob = action_probs.flatten()[action]
        return action, log_prob


# class MixerBlock(nn.Module):
#     """ Residual Block w/ token mixing and channel MLPs
#     Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
#     """
#     def __init__(
#             self,
#             dim,
#             seq_len,
#             mlp_ratio=(0.5, 4.0),


#     ):
#         super().__init__()
#         tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
#         self.norm1 = nn.LayerNorm(dim)
#         self.mlp_tokens = mlp_layer(seq_len, tokens_dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp_channels = mlp_layer(dim, channels_dim)

#     def forward(self, x):
#         x = x + self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
#         x = x + self.mlp_channels(self.norm2(x))
#         return x

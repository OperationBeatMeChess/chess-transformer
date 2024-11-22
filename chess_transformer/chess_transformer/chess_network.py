import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from adversarial_gym.chess_env import ChessEnv

from timm.layers import DropPath


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear1(self.norm1(x))
        out = self.linear2(self.gelu(out))
        out = self.dropout(out)
        out = self.norm2(out + x)
        return out


class StereoScaling(nn.Module):
    """ Learnable stereographic projection for each seq in attention map"""
    def __init__(self, dim: int):
        super().__init__()
        # Create the learnable x0 for stereographic projection
        self.x0 = nn.Parameter(torch.empty(1, 1, dim, 1))
        torch.nn.init.normal_(self.x0, std=0.02)

    def apply_stereo_scaling(self, x, x0, eps=1e-5):
        # Apply inverse stereographic projection to learnably scaled sphere
        s = (1 + x0) / (1 - x0 + eps)
        x_proj = 2 * x / (1 + s**2)
        return x_proj

    def forward( self, attn: torch.Tensor):
        attn = self.apply_stereo_scaling(attn, self.x0)
        return attn

 
class AttentionBlock(nn.Module):
    """ Multi-head self-attention module with optional 1D or 2D relative position bias.
     
    Using timm Swin Transformer implementation as a reference for the 2d relative position bias. The
    2D relative position bias is used in the Channel-Mixing Attention block and the 1D relative position
    bias is used in the Token-Mixing block. Bias is added to the attention scores

    Also uses a learnable stereographic projection as the scale in the attention calculation. In
    Transformers without Tears paper (I think), they mention learnable normalization may have benefit of
    smoothing the loss landscape.
    """
    def __init__(
            self,
            seq_len, 
            embed_dim, 
            num_heads, 
            dropout=0.0, 
            use_2d_relative_position=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = embed_dim
        self.seq_len = seq_len
        self.use_2d_relative_position = use_2d_relative_position
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.dim, "embed_dim must be divisible by num_heads"
    
        self.scale = StereoScaling(self.seq_len)
    
        # Splitting qkv into separate projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # Producing queries
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # Producing keys
        self.v_proj = nn.Linear(embed_dim, embed_dim)  # Producing values
        
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
        if self.use_2d_relative_position:
            self.h, self.w = self.compute_grid_dimensions(seq_len)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.h - 1) * (2 * self.w - 1), num_heads)
            )
            self.register_buffer("relative_position_index", self.get_2d_relative_position_index(self.h, self.w))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else: # 1D relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(2 * seq_len - 1, num_heads)
            )
            self.register_buffer("relative_position_index", self.get_1d_relative_position_index(seq_len))  
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def compute_grid_dimensions(self, n):
        """
        Compute grid dimensions (h, w) for 2D relative position bias. In our case, this will be the
        height and width of the chess board (8x8).
        """
        root = int(math.sqrt(n))
        for i in range(root, 0, -1):
            if n % i == 0:
                return (i, n // i)

    def get_2d_relative_position_index(self, h, w):
        """ Create pairwise relative position index for 2D grid."""

        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'))  # 2, h, w
        coords_flatten = coords.reshape(2, -1)  # 2, h*w
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
        relative_coords[:, :, 0] += h - 1  # Shift to start from 0
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1 
        relative_position_index = relative_coords.sum(-1)  # Shape: (h*w, h*w)
        return relative_position_index  # h*w, h*w

    def get_1d_relative_position_index(self, seq_len):
        # Compute relative position indices for 1D sequences
        coords = torch.arange(seq_len)
        relative_coords = coords[None, :] - coords[:, None]  # seq_len, seq_len
        relative_coords += seq_len - 1  # Shift to start from 0
        return relative_coords  # seq_len, seq_len

    def _get_rel_pos_bias(self):
        """Retrieve relative position bias based on precomputed indices for the attention scores."""
        # Retrieve and reshape the relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1).long()
        ]
        relative_position_bias = relative_position_bias.view(self.seq_len, self.seq_len, -1)  # seq_len, seq_len, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, seq_len, seq_len
        return relative_position_bias.unsqueeze(0)  # 1, num_heads, seq_len, seq_len

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.seq_len, f"Input sequence length {N} does not match expected {self.seq_len}"
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, N, C
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1))  # B, H, N, N
        
        relative_position_bias = self._get_rel_pos_bias() # 1, H, N, N
        attn = attn + relative_position_bias  # Broadcasting over batch size

        # attn *= self.scale
        attn = self.scale(attn)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = torch.matmul(attn, v)  # (B, H, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x

class MixerBlock(nn.Module):
    def __init__(self, piece_embed_dim, num_heads=16, dropout=0., drop_path=0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.channel_mixing_norm = nn.LayerNorm(piece_embed_dim)
        self.channel_mixing_attn = AttentionBlock(64,piece_embed_dim, num_heads, dropout=dropout, use_2d_relative_position=True)
        self.token_mixing_norm = nn.LayerNorm(piece_embed_dim)
        self.token_mixing_attn = AttentionBlock(piece_embed_dim, 64, 16, dropout=dropout, use_2d_relative_position=False)

        total_features = 64 * piece_embed_dim
        self.out_mlp = ResidualBlock(total_features, 2*total_features, dropout=dropout)

    def forward(self, x):
        # x shape: (B, 64, piece_embed)
        x = x + self.drop_path(self.token_mixing_attn(self.token_mixing_norm(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path(self.channel_mixing_attn(self.channel_mixing_norm(x)))
        x = self.out_mlp(x.view(x.size(0), -1)).view(x.size(0), 64, -1)
        return x


class ChessTransformer(nn.Module):
    """
    Creates a ChessNetwork that outputs a value and action for a given
    state/position using a Mixer-style network.
    
    The network processes the input state using an embedding for pieces and Mixer blocks,
    then feeds the output into separate prediction heads for policy, value, and critic outputs.
    """

    def __init__(self, device='cuda', num_mixer_layers=20, dropout=0.0):
        super().__init__()
        self.device = device
        self.action_dim = 4672
        action_embed_dim = 512
        piece_embed_dim=24
        self.action_embed = nn.Embedding(self.action_dim, action_embed_dim).to(device)

        num_features = 64  * piece_embed_dim

        # Embedding layer for pieces (integers ranging between -6 and 6)
        self.piece_embedding = nn.Embedding(num_embeddings=13, embedding_dim=piece_embed_dim).to(device)
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, piece_embed_dim))
        torch.nn.init.kaiming_normal_(self.pos_encoding, mode='fan_out', nonlinearity='relu')

        self.embed_mlp = ResidualBlock(num_features, 2 * num_features, dropout=dropout)

        # Mixer blocks
        # currently only params for num_heads, drop_path
        params_config = [(6, 0.05)] * 4 + [(8, 0.1)] * 4 + [(12, 0.15)] * 8 + [(24, 0.2)] * 4
        assert len(params_config) == num_mixer_layers, "Length of config does not match num_mixer_layers"
        
        self.mixer_layers = nn.Sequential(*[
            MixerBlock(piece_embed_dim, num_heads=params[0], drop_path=params[1])
            for params in params_config
        ]).to(device)

        print(f"Num mixer params: {sum(p.numel() for p in self.mixer_layers.parameters())}")

        # Policy head
        self.policy_head = nn.Sequential(
            ResidualBlock(num_features, 2*num_features),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, self.action_dim),
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            ResidualBlock(num_features, 2*num_features),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, 1),
            nn.Tanh()
        ).to(device)

        # Critic head
        critic_features = 64 * (piece_embed_dim + action_embed_dim//64)
        self.critic_head = nn.Sequential(
            ResidualBlock(critic_features, 2*critic_features),
            nn.Linear(critic_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, 1),
            nn.Tanh()
        ).to(device)
        
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()
        self.critic_loss = nn.MSELoss()
    
    def embed_state(self, x):
        x = x.view(x.size(0), -1)  # (B, 1, 8, 8) -> (B, 64)
        x = self.piece_embedding(x + 6)  # (B, 64) -> (B, 64, piece_embed_dim)
        x = x + self.pos_encoding.expand(x.size(0), -1, -1)
        x = self.embed_mlp(x.view(x.size(0), -1)).view(x.size(0), 64, -1)
        return x

    def forward(self, x, return_features=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.long, device=self.device)

        features = self.embed_state(x.long())  # Shape: (B, 64, piece_embed_dim)
        features = self.mixer_layers(features)  # Shape: (B, piece_embed_dim, 64)
        features = features.view(features.size(0), -1)

        action_logits = self.policy_head(features)
        board_val = self.value_head(features)

        if return_features:
            return action_logits, board_val, features
        else:  
            return action_logits, board_val
    
    def forward_critic(self, features, action):
        # Critic (S,A) -> R prediction
        action_inds = torch.argmax(action, dim=1)
        action_embed = self.action_embed(action_inds) # (B, action_embed_dim)
        critic_feats = torch.cat((features, action_embed), dim=-1) 
        critic_val = self.critic_head(critic_feats)
        return critic_val
    
    def get_action(self, state, legal_moves, sample_n=1):
        """ Randomly sample from top_n legal actions given input state"""

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        state = self.embed_state(state.long())
        features = self.mixer_layers(state)
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

        # sample from top-n policy prob indices
        top_n_indices = np.argpartition(action_probs_np, -top_n)[-top_n:]
        action = np.random.choice(top_n_indices)
        
        log_prob = action_probs.flatten()[action]
        return action, log_prob
    
# transformer = ChessTransformer().cuda()
# print(f"Num parameters: {sum(p.numel() for p in transformer.parameters())}")

# # Test the transformer with random inputs
# state = torch.randint(-1, 1, (32, 1, 8, 8)) * 6  # Random integer tensor between -6 and 6

# # # Forward pass
# action_logits, board_val, features = transformer(state.cuda(), return_features=True)

# # # # Test the critic head
# action = torch.randint(0, 4672, (32, 1))
# critic_val = transformer.forward_critic(features, action.cuda())

# # test relative position attention
# rel_pos = AttentionBlock(64,24, 8)
# x = torch.randn(32, 64, 24)
# times = []
# for i in range(200):
#     t1 = time.perf_counter()
#     out = rel_pos(x)
#     times.append(time.perf_counter()-t1)

# print(f"Time: {np.median(times)}")

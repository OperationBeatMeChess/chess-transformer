import random
from adversarial_gym.chess_env import ChessEnv
import torch
from torch import nn
import timm
import numpy as np

from torch.cuda.amp import GradScaler


class ChessNetworkSimple(nn.Module):
    """
    Creates an OBM ChessNetwork that outputs a value and action for a given
    state/position. 
    
    The network uses a feature extraction backbone from the Pytorch Image Model library (timm)
    and feeds the ouput of that into two separate prediction heads.

    The output of the policy network is a vector of size action_dim = 4762
    and the ouput of the value network is a single value.

    """

    def __init__(self, hidden_dim: int, device = 'cpu', base_lr = 0.0009, max_lr = 0.009):
        super().__init__()
        
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=False, 
                                                  img_size=8, patch_size=1, window_size=2, in_chans=1).to(device)
        self.hidden_dim = hidden_dim
        self.action_dim = 4672
        self.device = device

        self.grad_scaler = GradScaler()

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.swin_transformer.head.in_features, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.swin_transformer.head.in_features, hidden_dim),
            nn.RReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        ).to(device)
        
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, base_lr=base_lr, max_lr=max_lr)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            # Assuming its 8x8 array from chess env. Convert to (1,1,8,8) tensor
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        features = self.swin_transformer.forward_features(x).requires_grad_(True)
        action_logits = self.policy_head(features)
        board_val = self.value_head(features)
        return action_logits, board_val

    def to_action(self, action_logits, legal_moves, top_n):
        """ Randomly sample from top_n legal actions given output action logits """

        legal_actions = [ChessEnv.move_to_action(move) for move in legal_moves]

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

    def get_action(self, state, legal_moves, sample_n=1):
        """ Randomly sample from top_n legal actions given input state"""

        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device).requires_grad_(True)
        
        features = self.swin_transformer.forward_features(state).requires_grad_(True)
        features = features.view(features.shape[0], -1)
        
        policy_logits = self.policy_head(features)
        return self.to_action(policy_logits, legal_moves, top_n=sample_n)

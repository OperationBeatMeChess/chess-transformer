# import random
# from adversarial_gym.chess_env import ChessEnv

import torch
from torch import nn
import torch.nn.functional as F

import timm
import numpy as np

from torch.cuda.amp import GradScaler


def retrieve_relevant_memory(memory, topk_indices):
    """
    Retrieve the relevant memory slots based on topk indices for each batch item.

    :param memory: The memory tensor of shape (K, feature_size).
    :param topk_indices: The tensor of topk indices of shape (B, topk).
    :return: Tensor of relevant memory slots of shape (B, topk, feature_size).
    """
    B, topk = topk_indices.size()
    K, E = memory.size()

    # Expand the memory tensor to shape (B, K, feature_size) by adding a batch dimension
    expanded_memory = memory.unsqueeze(0).expand(B, K, E)

    # Reshape topk_indices to match the shape for gather
    # The new shape is (B, topk, 1), and then expand it to (B, topk, E) to match the expanded_memory shape
    topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, E)

    # Use advanced indexing to gather the relevant memory slots
    relevant_memory = expanded_memory.gather(1, topk_indices)

    return relevant_memory


class StateAttention(nn.Module):
    def __init__(self, query_dim, key_dim, topk):
        super().__init__()
        self.fc1 = nn.Linear(query_dim + key_dim, 512)
        self.fc2 = nn.Linear(512, 1)  # Output dimension is 1 for attention scoring
        self.topk = topk

    def forward(self, query, keys):
        # Expand keys to match the batch size of query: (B, K, E)
        # query.shape[0] is the batch size
        keys_expanded = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        
        # Expand query to match the number of keys: (B, 1, E) -> (B, K, E)
        query_expanded = query.unsqueeze(1).expand(-1, keys_expanded.size(1), -1)

        # Concatenate the expanded query with keys
        combined = torch.cat((query_expanded, keys_expanded), dim=2)

        # Pass through the MLP
        attn_scores = F.relu(self.fc1(combined))
        attn_scores = self.fc2(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        return attn_scores, attn_weights


class ActionResultAttention(nn.Module):
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.fc1 = nn.Linear(query_dim + key_dim, 256)  # Intermediate dimension
        self.fc2 = nn.Linear(256, 1)  # Output dimension is 1 for attention scoring

        self.feature_size = query_dim

    def forward(self, query, relevant_memory):
        # Extract state features, actions, results from relevant_memory
        memory_state_features = relevant_memory[..., :self.feature_size] # [B, topk, feature_size]
        actions               = relevant_memory[..., self.feature_size:-1].long()  # embedded action vector
        results               = relevant_memory[..., -1]  # [B, topk]
        
        # # Create masks for wins (1.0), draws (0.0), and losses (-1.0)
        # win_mask  = (relevant_memory[..., -1] == 1.0).float()   # Shape: (B, topk)
        # draw_mask = (relevant_memory[..., -1] ==  0.0).float()   # Shape: (B, topk)
        # loss_mask = (relevant_memory[..., -1] ==  -1.0).float()   # Shape: (B, topk)
        
        # Expand query to match the dimensions of state features
        query_expanded = query.unsqueeze(1).expand(-1, memory_state_features.size(1), -1) # (B, topk, feature_size)
        
        # Concatenate the expanded query with state features
        combined = torch.cat((query_expanded, memory_state_features), dim=2)
        
        # Pass through the MLP
        attn_scores = F.gelu(self.fc1(combined))
        attn_scores = self.fc2(attn_scores).squeeze(-1)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # # Weight the masks by attention weights
        # weighted_win_mask = win_mask * attn_weights
        # weighted_draw_mask = draw_mask * attn_weights
        # weighted_loss_mask = loss_mask * attn_weights

        # # Sum across topk to get the count for each category
        # win_count = weighted_win_mask.sum(dim=1).unsqueeze(-1)   # Shape: (B, 1)
        # draw_count = weighted_draw_mask.sum(dim=1).unsqueeze(-1) # Shape: (B, 1)
        # loss_count = weighted_loss_mask.sum(dim=1).unsqueeze(-1) # Shape: (B, 1)

        # # Concatenate to form the final output
        # result_counts = torch.cat([win_count, draw_count, loss_count], dim=1)  # Shape: (B, 3)

        # Compute weighted sum of action embeddings
        weighted_actions = torch.sum(actions * attn_weights.unsqueeze(-1), dim=1) # (B,embed_action_dim)
        
        # Compute weighted sum of results
        weighted_results = torch.sum(results * attn_weights, dim=1).unsqueeze(-1) # (B,1)

        # return result_counts, weighted_actions
        return weighted_results, weighted_actions

class MultiActionEmbedding(nn.Module):
    def __init__(self, num_actions, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(torch.randn(num_actions, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self, probability_vectors):
        # Multiply probability vectors by embedding matrix
        # This step computes a weighted sum of embeddings for each binary encoding or probabilty
        embedded = torch.matmul(probability_vectors, self.embeddings)

        # The result is a batch of vectors where each vector is a weighted sum of embeddings
        return embedded

class WriteHead(nn.Module):
    
    def __init__(self, memory_size, feature_size, reduced_feature_size=750):
        super().__init__()
        self.memory_size = memory_size  # K - Total number of memory slots
        self.feature_size = feature_size  # Size of the feature vector
        
        self.threshold = nn.Parameter(torch.tensor(0.05))

        # # Linear layer to project features into a smaller common space
        #
        # TODO: test if reduce works later
        # reduced_feature_size = feature_size
        # self.input_projection = nn.Linear(feature_size, reduced_feature_size)
        # self.memory_projection = nn.Linear(feature_size, reduced_feature_size)

        # MLP to compute attention scores
        self.attention_mlp = nn.Sequential(
            nn.Linear(feature_size*2, feature_size),  # Combining input and memory features
            nn.Tanh(),
            nn.Linear(feature_size, 1)
        )

    def forward(self, input_data, memory):
        """
        input_data: a batch of data points to be written, shaped (batch_size, feature_size)
        memory: the current state of the memory, shaped (memory_size, feature_size)
        """
        updated_memory = memory.clone()

        for i in range(input_data.size(0)):
            # Expand the single input data point for comparison with each memory slot
            expanded_input = input_data[i].unsqueeze(0).expand(self.memory_size, -1)

            # Compute attention scores for this single input data point
            combined_features = torch.cat((expanded_input, memory), dim=1)
            attention_scores = self.attention_mlp(combined_features).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=0)

            # Determine the best memory slot for this input data point
            mask = torch.gt(attention_weights, self.threshold)
            best_slot = attention_weights.argmax()

            # Update the memory slot if the condition is met
            if mask[best_slot]:
                updated_memory[best_slot] = input_data[i]

        return updated_memory
    
    # def forward(self, input_data, memory):
    #     """
    #     NOTE: faster but more memory
    #     input_data: a batch of data points to be written, shaped (batch_size, feature_size)
    #     memory: the current state of the memory, shaped (memory_size, feature_size)
    #     """
    #     # Project into the common (smaller) space before doing MLP attention
    #     # projected_input = self.input_projection(input_data)
    #     # projected_memory = self.memory_projection(memory)
        
    #     # Expand dimensions to prepare for broadcasting
    #     expanded_input = input_data.unsqueeze(1).expand(-1, self.memory_size, -1)
    #     expanded_memory = memory.unsqueeze(0).expand(input_data.size(0), -1, -1)
        
    #     # Concatenate the input with each memory slot's features
    #     combined_features = torch.cat((expanded_input, expanded_memory), dim=2)
        
    #     # Compute attention scores using the MLP
    #     attention_scores = self.attention_mlp(combined_features).squeeze(-1)
        
    #     # Apply softmax to get a probability distribution over the memory slots
    #     attention_weights = F.softmax(attention_scores, dim=1)

    #     # Find the best slots based on max attention weights.
    #     _, best_slots = attention_weights.max(dim=1)
        
    #     # Create a mask based on the threshold.
    #     mask = torch.gt(attention_weights, self.threshold)

    #     # Vectorized memory update
    #     update_mask = torch.zeros_like(memory[:, 0], dtype=torch.bool)
    #     update_mask[best_slots[mask]] = True
    #     data_to_update = input_data[mask]
    #     updated_memory = memory.clone()
    #     updated_memory[update_mask] = data_to_update

    #     return updated_memory


class ReadHead(nn.Module):
    """ 
    ReadHead module for SuperChessNetwork. 
    
    This module takes the current position as a query and retrieves relevant data in memory. 

    """
    def __init__(self, state_feature_size, topk):
        super().__init__()
        self.relevance_attention = StateAttention(state_feature_size, state_feature_size, topk)
        self.eval_attention = ActionResultAttention(state_feature_size, state_feature_size) 

        self.state_feature_size = state_feature_size
        self.topk = topk

    def forward(self, state_features, memory):
        # Get state features from memory
        feature_memory = memory[:, :self.state_feature_size] # [K,feature_size]

        # First stage of attention: determine relevant memory slots
        attn_scores, attn_weights = self.relevance_attention(state_features, feature_memory)
        _, topk_slot_idxs = torch.topk(attn_weights, self.topk, dim=1)

        # Retrieve relevant memory using topk indices
        relevant_memory = retrieve_relevant_memory(memory, topk_slot_idxs) # [B, topk, memory_feature_size]

        # Second stage of attention: determine values
        # Return weighted sum of results and weighted frequencies of actions
        weighted_results, weighted_actions = self.eval_attention(state_features, relevant_memory)

        return weighted_results, weighted_actions

 
class ValueMLPAttention(nn.Module):
    def __init__(self, query_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, 512),  # Adjust the dimensions as needed
            nn.GELU(),
            nn.Linear(512, 1)  # Ensure the input dimension here matches the previous layer's output
        )

    def forward(self, query, weighted_results, board_val):
        # Pass the query (state features) through the MLP
        attn_weight = torch.sigmoid(self.mlp(query))  # Apply sigmoid here

        # Calculate the weighted average
        combined_results = attn_weight * weighted_results + (1 - attn_weight) * board_val

        return combined_results

    
class SuperChessNetwork(nn.Module):
    """
    Creates an OBM ChessNetwork that outputs a value and action for a given
    state/position. 
    
    The network uses a feature extraction backbone from the Pytorch Image Model library (timm)
    and feeds the ouput of that into two separate prediction heads.

    The output of the policy network is a vector of size action_dim = 4762
    and the ouput of the value network is a single value.

    """

    def __init__(self, memory_size, topk=1000, device = 'cpu', base_lr = 0.0009, max_lr = 0.009):
        super().__init__()
        
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=False,
                                                  img_size=8, patch_size=1,
                                                  window_size=2, in_chans=1).to(device)
        
        feature_size = self.swin_transformer.head.in_features

        self.swin_transformer.fc = nn.Identity()

        self.action_dim = 4672
        self.action_embedding_dim = 128
        self.device = device


        self.grad_scaler = GradScaler()

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(feature_size + self.action_embedding_dim, 768),
            nn.GELU(),
            nn.Linear(768, self.action_dim),
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Tanh()
        ).to(device)

        self.action_embedding = MultiActionEmbedding(self.action_dim, self.action_embedding_dim)
        self.read_head = ReadHead(feature_size,  topk=topk)
        self.write_head = WriteHead(memory_size, feature_size + self.action_embedding_dim + 1)
        self.mlp_attention = ValueMLPAttention(feature_size)
        
        self.register_buffer('memory', torch.zeros(memory_size, feature_size + self.action_embedding_dim + 1, device='cpu'))
     
        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()
        self.mem_policy_loss =  nn.CosineSimilarity()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=base_lr)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50_000)
    
    # def memory_loss(self, weighted_actions, weighted_results, gt_actions, gt_results):
    #     # Convert ground truth actions to embeddings
    #     gt_action_embeddings = self.action_embedding(gt_actions)  # Assuming gt_actions are (B, 4672)

    #     mem_value_loss = self.value_loss(weighted_results.squeeze(), gt_results)
    #     mem_policy_loss = self.value_loss(weighted_actions, gt_action_embeddings) # MSE loss for now
    #     return mem_policy_loss, mem_value_loss
    
    def write_to_memory(self, features, action, result):
        # Concat features. Action is a one-hot vector and result is a single value
        action_embedding = self.action_embedding(action)
        features = torch.cat((features, action_embedding, result.unsqueeze(-1)), dim=1)
        self.memory = self.write_head(features, self.memory)
         
    def initialize_memory(self, train_loader):
        # Initialize memory with random data
        mem_idx = 0
        for state, action, result in train_loader:
            if mem_idx >= self.memory.size(0): break
            
            state, action, result = state.to(self.device), action.to(self.device), result.to(self.device)
            with torch.inference_mode():
                features = self.swin_transformer.forward_features(state.float().unsqueeze(1)).squeeze()
                action_embedding = self.action_embedding(action)
                combined_features = torch.cat((features, action_embedding, result.unsqueeze(-1)), dim=1)

                write_count = min(combined_features.size(0), self.memory.size(0) - mem_idx)
                self.memory[mem_idx:mem_idx + write_count].copy_(combined_features[:write_count])
                mem_idx += write_count


    def forward(self, x):
        if isinstance(x, np.ndarray):
            # Assuming its 8x8 array from chess env. Convert to (1,1,8,8) tensor
            x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0).unsqueeze(0)
        
        features = self.swin_transformer.forward_features(x).squeeze()

        # Read from memory
        weighted_results, weighted_actions = self.read_head(features, self.memory)

        # Combine features with memory features
        policy_features = torch.cat((features, weighted_actions), dim=1)

        action_logits = self.policy_head(policy_features)
        board_vals = self.value_head(features)

        # # Do MLP attention to get a weighted average of memory results and prediction results
        # # Concatenate the two results to use as key/value in attention
        # results_combined = torch.cat((weighted_results, board_vals), dim=1)
        
        # Apply MLP attention
        value_output = self.mlp_attention(features, weighted_results, board_vals)

        return action_logits, board_vals, value_output, (features, weighted_actions, weighted_results)
    


# ############### TEST CODE FOR READ HEAD
# # Instantiate and use the MLPAttention class
# # Parameters
# query_dim = 1000
# key_dim = 5673
# topk = 5
# batch_size = 32
# memory_size = 100

# # Simulated input data
# query = torch.rand(batch_size, query_dim)
# memory = torch.rand(memory_size, key_dim)  # Assuming 1000 key features + 2 additional features

# mlp_attn = ReadHead(query_dim, topk)
# weighted_results, weighted_actions = mlp_attn(query, memory)
# print(weighted_results.shape)
# print(weighted_actions.shape)
# ############### TEST CODE FOR READ HEAD


# ############## TEST CODE FOR WRITE HEAD
# # Define the sizes
# batch_size = 2  # Number of data points in the input batch
# memory_size = 3  # Total number of memory slots
# # feature_size = 1002  # Size of the feature vector
# feature_size = 5  # Size of the feature vector

# # Instantiate the WriteHead module
# write_head = WriteHead(memory_size, feature_size)

# # Generate some random data for the input data and memory
# input_data = torch.randn(batch_size, feature_size)
# memory = torch.zeros(memory_size, feature_size)

# # Forward pass through the write head
# print(memory)
# updated_memory = write_head(input_data, memory)
# print(updated_memory)  # Should be (memory_size, feature_size)
# ############## TEST CODE FOR WRITE HEAD

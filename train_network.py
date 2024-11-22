from collections import OrderedDict
import time, os, random
import wandb

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from chess_transformer.dataset.chess_dataset import ChessDataset
from chess_transformer.chess_transformer.chess_network import ChessTransformer
from chess_transformer.chess_transformer.warmup_torch import WarmupLR


PGN_DIR_TRAIN = '/home/kage/chess_workspace/PGN_dataset/Dataset/train'
PGN_DIR_TEST = '/home/kage/chess_workspace/PGN_dataset/Dataset/test'

MODEL_PATH = "/home/kage/chess_workspace/chessAttnMixer20-rpe-lrup.pt"
MODEL_SAVEPATH = "/home/kage/chess_workspace/chessAttnMixer20-rpe-lrup.pt"

NUM_ROUNDS = 50
DATASET_SIZE_TRAIN = 500
DATASET_SIZE_TEST = 30
NUM_THREADS = 20

BATCH_SIZE_TRAIN = 2475
BATCH_SIZE_TEST = 2475

WANDB = True
LOG_EVERY = 200
VALIDATION_EVERY = 15_000


class RunningAverage:
    def __init__(self):
        self.counts = {}
        self.averages = {}

    def add(self, var_names):
        var_names = [var_names] if isinstance(var_names, str) else var_names
        for var_name in var_names:
            if var_name not in self.averages:
                self.averages[var_name] = 0.0
                self.counts[var_name] = 0

    def update(self, var_name, value=None):
        if isinstance(var_name, dict):
            for k, v in var_name.items():
                if k not in self.averages:
                    print(f"Variable {k} is not being tracked. Use add method to track.")
                    continue
                self.update(k, v)
        else:
            if var_name not in self.averages:
                print(f"Variable {var_name} is not being tracked. Use add method to track.")
                return
            self.counts[var_name] += 1
            self.averages[var_name] += (value - self.averages[var_name]) / self.counts[var_name]

    def get_average(self, var_names):
        if isinstance(var_names, str):
            return self.averages.get(var_names, None)

        return {var_name: self.averages.get(var_name, None) for var_name in var_names}

    def reset(self, var_names=None):
        if var_names is None:
            self.counts = {}
            self.averages = {}
        else:
            var_names = [var_names] if isinstance(var_names, str) else var_names
            for var_name in var_names:
                if var_name in self.averages:
                    self.counts[var_name] = 0
                    self.averages[var_name] = 0.0
                else:
                    print(f"Variable {var_name} is not being tracked.")


def run_validation(model, stats):
    test_data = [pgn.path for pgn in os.scandir(PGN_DIR_TEST) if pgn.name.endswith(".pgn")]
    sampled_test_data = random.sample(test_data, DATASET_SIZE_TEST)
    t1 = time.perf_counter()
    test_dataset = ChessDataset(sampled_test_data, load_parallel=True, num_threads=NUM_THREADS)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)   
    
    print(f"Loaded validation dataset with {len(test_dataset)} positions - {time.perf_counter()-t1} seconds")

    model.eval()
    stats.reset([
        "val_loss", 
        "val_ploss", 
        "val_vloss",
        "val_closs"
    ])
    
    t1 = time.perf_counter()
    with torch.no_grad():
        for i, (state, action, result) in enumerate(val_loader):
            state = state.float().to('cuda')
            action = action.to('cuda')
            result = result.float().to('cuda')
            
            policy_output, value_output, features = model(state.unsqueeze(1), return_features=True)
            critic_output = model.forward_critic(features, action.long())
                        
            policy_loss = model.policy_loss(policy_output.squeeze(), action)
            value_loss = model.value_loss(value_output.squeeze(), result)
            critic_loss = model.critic_loss(critic_output.squeeze(), result)
            
            loss = policy_loss + value_loss

            stats.update({
                "val_loss": loss.detach().item(),
                "val_ploss": policy_loss.detach().item(),
                "val_vloss": value_loss.detach().item(),
                "val_closs": critic_loss.detach().item(),
            })
        
    return stats.get_average('val_loss'), stats.get_average('val_ploss'), stats.get_average('val_vloss'), stats.get_average('val_closs')


def training_round(
        model, 
        train_loader, 
        optimizer,
        scheduler,
        grad_scaler,
        num_epochs=10, 
        log_every=1000, 
        validation_every=20_000):
    
    stats = RunningAverage()
    stats.add([
        "train_loss", 
        "val_loss", 
        "train_ploss", 
        "train_vloss", 
        "train_closs",
        "val_ploss", 
        "val_vloss",
        "val_closs"
    ])

    best_val_loss = 1000

    for epoch in range(num_epochs): 
        model.train()
        t1 = time.perf_counter()
        
        for i, (state, action, result) in enumerate(train_loader):
            state = state.float().to('cuda')
            action = action.to('cuda')
            result = result.float().to('cuda')
            
            # AMP with gradient clipping and lr scheduler
            with torch.amp.autocast('cuda'):
                policy_output, value_output, features = model(state.unsqueeze(1), return_features=True)
                critic_output = model.forward_critic(features, action.long())
                
                policy_loss = model.policy_loss(policy_output.squeeze(), action)
                value_loss = model.value_loss(value_output.squeeze(), result)
                critic_loss = model.critic_loss(critic_output.squeeze(), result)
                
                loss = policy_loss + value_loss + critic_loss

            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            scale = grad_scaler.get_scale()
            grad_scaler.update()
            skip_lr_sched = scale > grad_scaler.get_scale()
            
            if not skip_lr_sched: scheduler.step()

            # torch.cuda.synchronize()
            stats.update({
                "train_loss": loss.item(),
                "train_ploss": policy_loss.item(),
                "train_vloss": value_loss.item(),
                "train_closs": critic_loss.item()
                })
            
            if i % log_every == 0 and i > 0 and WANDB:
                wandb.log({
                    "lr": scheduler.get_last_lr()[0], 
                    "train_loss": stats.get_average('train_loss'), 
                    "train_ploss": stats.get_average('train_ploss'), 
                    "train_vloss": stats.get_average('train_vloss'), 
                    "train_closs": stats.get_average('train_closs'),
                    "iter": i
                })
            
            if i % validation_every == 0 and i > 0 :
                val_loss, val_ploss, val_vloss, val_closs = run_validation(model, stats)
                
                if WANDB:
                    wandb.log({
                        "val_loss": val_loss, 
                        "val_ploss": val_ploss, 
                        "val_vloss": val_vloss, 
                        "val_closs": val_closs,
                        "iter": i
                    })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), MODEL_SAVEPATH)
            
        print(f"Epoch took {time.perf_counter()-t1} seconds ")
        torch.save(model.state_dict(), MODEL_SAVEPATH)


def run_training(
        model, 
        optimizer,
        scheduler,
        num_rounds
):
    
    train_data = [pgn.path for pgn in os.scandir(PGN_DIR_TRAIN) if pgn.name.endswith(".pgn")]

    grad_scaler = torch.amp.GradScaler("cuda")

    for round in range(num_rounds):
        print(f"Starting round {round}")
        # build dataset 
        # randomly sample dataset_size pgn files 
        t1 = time.perf_counter()
        sampled_train_data = random.sample(train_data, DATASET_SIZE_TRAIN)
        train_dataset = ChessDataset(sampled_train_data, load_parallel=True, num_threads=NUM_THREADS)
        print(f"Successfully loaded dataset with {len(train_dataset)} images - {time.perf_counter()-t1} seconds")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
        
        training_round(
            model, 
            train_loader, 
            optimizer, 
            scheduler, 
            grad_scaler, 
            num_epochs=1, 
            log_every=LOG_EVERY, 
            validation_every=VALIDATION_EVERY)

        del train_dataset
        del train_loader


def align_state_dict_keys(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Add the specified prefix to each key
        new_key = prefix + key
        new_state_dict[new_key] = value
    return new_state_dict

if __name__ == "__main__":
    model = ChessTransformer()
    model.to('cuda')
    model = torch.compile(model)

    print(f"No. of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000095)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.5, 100_000)
    # scheduler = WarmupLR(scheduler, init_lr=0.00001, num_warmup=1000, warmup_strategy='cos')

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        # checkpoint = align_state_dict_keys(torch.load(MODEL_PATH))
        model.load_state_dict(checkpoint)
        print("Loaded model from checkpoint")

    if WANDB:
        wandb.init(project="chess-transformer", id='bbbgnwot', resume='must')

    run_training(model, optimizer, scheduler, NUM_ROUNDS)


from torch.multiprocessing import Pool, set_start_method
import chess
import chess.pgn

import torch
from numba import jit
from torch.utils.data import Dataset

from adversarial_gym.chess_env import ChessEnv
from stockfish import Stockfish


def create_sparse_vector(action_probs):
    # Initialize a list of zeros for all possible actions
    sparse_vector = [0.] * 4672
    
    for action, prob in action_probs.items():
        sparse_vector[action] = prob
    return sparse_vector


class ChessDataset(Dataset):
    def __init__(self, pgn_files, load_parallel=False, num_threads=0):
        super().__init__()
        self.pgn_files = pgn_files
        self.data = self.generate_pgn_data() if not load_parallel else self.generate_pgn_data_parallel(num_threads=num_threads)
        
    def generate_pgn_data(self):
        """ Return a list of tuples with (state, action, result) """
        data = []
        for pgn_file in self.pgn_files:
            print(f"Processing {pgn_file}")
            with open(pgn_file) as pgn:
                while (game := chess.pgn.read_game(pgn)) is not None:
                    state_maps, actions, results = self.get_game_data(game)
                    game_data = list(zip(state_maps, actions, results))
                    data.extend(game_data)
        return data
    
    def generate_pgn_data_single(self, pgn_file):
        data = []
        with open(pgn_file) as pgn:
            while (game := chess.pgn.read_game(pgn)) is not None:
                state_maps, actions, results = self.get_game_data(game)
                game_data = list(zip(state_maps, actions, results))
                data.extend(game_data)
        return data
    
    def generate_pgn_data_parallel(self, num_threads):
        try:
            # Set the start method to 'spawn'
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 'spawn' is already set

        with Pool(num_threads) as pool:
            results = pool.map(self.generate_pgn_data_single, self.pgn_files)

        return [item for sublist in results for item in sublist]
    
    def get_game_data(self, game):
        """
        Generate a game's training data, where X = board_state and Y1 = action taken, Y2 = result.
        A move is given by [from_square, to_square, promotion, drop]
        """
        states = []
        actions = []
        results = []

        board = game.board()
        result = game.headers['Result']
        result = self.result_to_number(result)
        for move in game.mainline_moves():
            canon_state = self.get_canonical_state(board, board.turn)
            action = ChessEnv.move_to_action(move)

            states.append(canon_state)
            actions.append(action)
            results.append(result*-1 if board.turn==0 else result) # flip value to match canonical state
            board.push(move)
        
        if game.errors: print(game.errors, game.headers)
        
        return states, actions, results
     
    def get_canonical_state(self, board, turn):
        state = ChessEnv.get_piece_configuration(board)
        state = state if turn else -state
        return state

    def result_to_number(self, result):
        if result == "1-0": return 1
        if result == "0-1": return -1
        else: return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, result = self.data[idx]
        action_probs = create_sparse_vector({action: 1.0})
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        return state, action_probs, result

class LegalActionChessDataset(ChessDataset):
    def generate_pgn_data(self):
        """ Return a list of tuples with (state, action, result) """
        data = []
        for pgn_file in self.pgn_files:
            print(f"Processing {pgn_file}")
            with open(pgn_file) as pgn:
                while (game := chess.pgn.read_game(pgn)) is not None:
                    state_maps, actions, results, legal_actions = self.get_game_data(game)
                    game_data = list(zip(state_maps, actions, results, legal_actions))
                    data.extend(game_data)
        return data
    
    def generate_pgn_data_single(self, pgn_file):
        data = []
        with open(pgn_file) as pgn:
            while (game := chess.pgn.read_game(pgn)) is not None:
                state_maps, actions, results, legal_actions = self.get_game_data(game)
                game_data = list(zip(state_maps, actions, results, legal_actions))
                data.extend(game_data)
        return data
    
    def generate_pgn_data_parallel(self, num_threads):
        try:
            # Set the start method to 'spawn'
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 'spawn' is already set

        with Pool(num_threads) as pool:
            results = pool.map(self.generate_pgn_data_single, self.pgn_files)

        return [item for sublist in results for item in sublist]
    
    def get_game_data(self, game):
        """
        Generate a game's training data, where X = board_state and Y1 = action taken, Y2 = result.
        A move is given by [from_square, to_square, promotion, drop]
        """
        states = []
        actions = []
        results = []
        legal_actions = []

        board = game.board()
        result = game.headers['Result']
        result = self.result_to_number(result)
        for move in game.mainline_moves():
            canon_state = self.get_canonical_state(board, board.turn)
            action = ChessEnv.move_to_action(move)

            legal_actions.append([ChessEnv.move_to_action(a) for a in board.legal_moves])
            states.append(canon_state)
            actions.append(action)
            results.append(result*-1 if board.turn==0 else result) # flip value to match canonical state
            board.push(move)
        
        if game.errors: print(game.errors, game.headers)
        
        return states, actions, results, legal_actions

    def __getitem__(self, idx):
        state, action, result, legal_actions = self.data[idx]
        action_probs = create_sparse_vector({action: 1.0})
        legal_actions = create_sparse_vector({a: 1.0 for a in legal_actions})
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        legal_actions = torch.tensor(legal_actions, dtype=torch.float32)
        return state, action_probs, result, legal_actions
    


class TestChessDataset(Dataset):
    def __init__(self, pgn_files, load_parallel=False, num_threads=0):
        super().__init__()
        self.pgn_files = pgn_files
        self.data = self.generate_pgn_data() if not load_parallel else self.generate_pgn_data_parallel(num_threads=num_threads)
        
    def generate_pgn_data(self):
        """ Return a list of tuples with (state, action, result) """
        data = []
        for pgn_file in self.pgn_files:
            with open(pgn_file) as pgn:
                while (game := chess.pgn.read_game(pgn)) is not None:
                    state_maps, actions, results = self.get_game_data(game)
                    game_data = list(zip(state_maps, actions, results))
                    data.extend(game_data)
        return data
    
    def generate_pgn_data_single(self, pgn_file):
        data = []
        with open(pgn_file) as pgn:
            while (game := chess.pgn.read_game(pgn)) is not None:
                state_maps, actions, results = self.get_game_data(game)
                game_data = list(zip(state_maps, actions, results))
                data.extend(game_data)
        return data
    
    def generate_pgn_data_parallel(self, num_threads):
        try:
            # Set the start method to 'spawn'
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 'spawn' is already set

        with Pool(num_threads) as pool:
            results = pool.map(self.generate_pgn_data_single, self.pgn_files)

        return [item for sublist in results for item in sublist]
    
    def get_game_data(self, game):
        """
        Generate a game's training data, where X = board_state and Y1 = action taken, Y2 = result.
        A move is given by [from_square, to_square, promotion, drop]
        """
        states = []
        actions = []
        results = []

        board = game.board()
        result = game.headers['Result']
        result = self.result_to_number(result)
        for move in game.mainline_moves():
            canon_state = self.get_canonical_state(board, board.turn)
            action = ChessEnv.move_to_action(move)

            states.append(canon_state)
            actions.append(action)
            results.append(result*-1 if board.turn==0 else result) # flip value to match canonical state
            board.push(move)
        
        if game.errors: print(game.errors, game.headers)
        
        return states, actions, results
     
    def get_canonical_state(self, board, turn):
        state = ChessEnv.get_piece_configuration(board)
        state = state if turn else -state
        return state

    def result_to_number(self, result):
        if result == "1-0": return 1
        if result == "0-1": return -1
        else: return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, result = self.data[idx]
        action_probs = create_sparse_vector({action: 1.0})
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        return state, action_probs, result

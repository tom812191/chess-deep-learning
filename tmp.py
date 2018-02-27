from keras.models import load_model

from agent.mcts import ChessMonteCarloTreeSearch
from chess_environment.game_parser import ChessPositionParser

import config


# Initialize the configuration
cfg = config.Config()

# Load up the trained model
model = load_model(cfg.resources.best_model_path)
pp = ChessPositionParser(cfg)

mcts = ChessMonteCarloTreeSearch(cfg, model, pp, tau=None)
mcts.search_moves().calc_move_probabilities()

print(mcts.move_probabilities)
print(mcts.get_next_move())

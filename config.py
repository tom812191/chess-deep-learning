import os
from enum import Enum

from util import uci_util


class Config:
    labels = sorted(uci_util.all_uci_labels())
    n_labels = int(len(labels))
    piece_index_map = {'KQRBNPkqrbnp'[i]: i for i in range(12)}
    castling_index_map = {'KQkq'[i]: i for i in range(4)}

    show_warnings = False

    def __init__(self):
        self.policy_model = PolicyModelConfig()
        self.move_probability_model = MoveProbabilityModelConfig()
        self.resources = ResourceConfig()
        self.trainer = TrainerConfig()
        self.play = PlayerConfig()
        self.training_type = TrainingType


class PolicyModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 256

    # 8x8 board with 20 channels
    # The 18 channels are:
    #     12 for piece type: KQRBNPkqrbnp
    #      4 for castling KQkq
    #      1 for en passant
    #      1 for elo
    input_shape = (18, 8, 8)


class MoveProbabilityModelConfig:
    dense_layer_sizes = (256, 256)
    l2_reg = 1e-4

    # 101 node inputs are:
    #      20 for prior move probabilities (ordered most likely to least likely)
    #      80, 4 for each move that represent stockfish value at various depths after each move
    #      1 for player elo rating
    num_candidate_moves = 20
    valuation_depths = [1, 5, 10]
    input_size = num_candidate_moves * (len(valuation_depths) + 1) + 1


class ResourceConfig:
    # data_directory = os.path.abspath('/home/ubuntu/data/chess-deep-learning')
    data_directory = os.path.abspath('/Users/tom/Projects/Portfolio/data/chess-deep-learning')
    lichess_download_list = os.path.join(data_directory, 'lichess_download_list.txt')
    lichess_download_list_cv = os.path.join(data_directory, 'lichess_download_list_cv.txt')
    download_chunk_size = 10 * 1024  # 1024 * 1024 * 8  # 8 MB
    training_data_directory = os.path.join(data_directory, 'training')

    best_model_path = os.path.join(data_directory, 'best_model.hdf5')
    best_move_model_path = os.path.join(data_directory, 'best_move_model.hdf5')
    database_settings_file = os.path.join(data_directory, 'db_settings.json')

    stockfish_path = os.path.join(data_directory, 'stockfish')


class TrainingType(Enum):
    FILE = 1
    DATABASE = 2


class TrainerConfig:
    # one of (FILE, DATABASE)
    train_type = TrainingType.DATABASE

    continue_from_best = False

    # Don't use convention that a single epoch is an iteration over all data. We'll want to save model more frequently.
    training_examples = 1863981
    cv_examples = 194063

    batch_size = 2048

    steps_per_epoch = training_examples // batch_size
    steps_per_epoch_cv = cv_examples // batch_size

    epochs = 100

    min_position_visits_total = 6
    min_position_visits = 3

    batch_size_moves = 64
    steps_per_epoch_moves = 10
    steps_per_epoch_moves_cv = 1
    epochs_moves = 10


class PlayerConfig:
    num_simulations = 1200
    upper_confidence_tree_score_constant = 1.0

    mcts_cache_size = 10000
    mcts_cache_path = '/users/tom/tmp/cache.p'

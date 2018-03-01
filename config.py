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
        self.model = ModelConfig()
        self.resources = ResourceConfig()
        self.trainer = TrainerConfig()
        self.play = PlayerConfig()
        self.training_type = TrainingType


class ModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 256
    distributed = True

    # 8x8 board with 20 channels
    # The 18 channels are:
    #     12 for piece type: KQRBNPkqrbnp
    #      4 for castling KQkq
    #      1 for en passant
    #      1 for elo
    input_shape = (18, 8, 8)


class ResourceConfig:
    # data_directory = os.path.abspath('/home/ubuntu/data/chess-deep-learning')
    data_directory = os.path.abspath('/Users/tom/Projects/Portfolio/data/chess-deep-learning')
    lichess_download_list = os.path.join(data_directory, 'lichess_download_list.txt')
    lichess_download_list_cv = os.path.join(data_directory, 'lichess_download_list_cv.txt')
    download_chunk_size = 10 * 1024  # 1024 * 1024 * 8  # 8 MB
    training_data_directory = os.path.join(data_directory, 'training')

    best_model_path = os.path.join(data_directory, 'best_model.hdf5')
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
    total_examples = 830469
    training_examples = 9 * total_examples / 10
    cv_examples = total_examples / 10

    batch_size = 2048

    steps_per_epoch = training_examples // batch_size
    steps_per_epoch_cv = cv_examples // batch_size

    epochs = 5

    min_position_visits_total = 10
    min_position_visits = 5


class PlayerConfig:
    num_simulations = 1200
    upper_confidence_tree_score_constant = 1.0

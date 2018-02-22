import os

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


class ModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 256
    distributed = True

    # 8x8 board with 20 channels
    # The 20 channels are:
    #     12 for piece type: KQRBNPkqrbnp
    #      4 for castling KQkq
    #      1 for en passant
    #      2 for white elo, black, elo
    #      1 for time control
    input_shape = (20, 8, 8)


class ResourceConfig:
    # data_directory = os.path.abspath('/home/ubuntu/data/chess-deep-learning')
    data_directory = os.path.abspath('/Users/tom/Projects/Portfolio/data/chess-deep-learning')
    lichess_download_list = os.path.join(data_directory, 'lichess_download_list.txt')
    lichess_download_list_cv = os.path.join(data_directory, 'lichess_download_list_cv.txt')
    download_chunk_size = 10 * 1024  # 1024 * 1024 * 8  # 8 MB

    best_model_path = os.path.join(data_directory, 'best_model.hdf5')
    database_settings_file = os.path.join(data_directory, 'db_settings.json')


class TrainerConfig:
    loss_weights = [1.25, 1.0]

    batch_size = 1024
    steps_per_epoch = 100
    steps_per_epoch_cv = 1

    continue_from_best = True

    use_curriculum = True
    curriculum = (
        {
            'query_conditions': (
                'white_elo > 2000',
                'black_elo > 2000',
                'has_checkmate',
                "file_id = 'MLBASE'",
            ),
            'approx_rows': 50000
        },

    )

    # Don't use convention that a single epoch is an iteration over all data. We'll want to save model more frequently.

    epochs = sum([c['approx_rows'] for c in curriculum]) * 40 * 2 // (batch_size * steps_per_epoch)

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Dense
from keras.regularizers import l2

import config


class ChessMoveModel:
    """
    The feed forward artificial neural network that will take prior move probabilities, move strengths (as
    determined by stockfish), and player strength to move probabilities
    """
    def __init__(self, cfg: config.Config):
        self.model = None
        self.config = cfg

    def build_model(self):
        """
        Build the Keras model
        """
        model_config = self.config.move_probability_model
        input_layer = model = Input((model_config.input_size,))

        # Build dense layers
        for index, size in enumerate(model_config.dense_layer_sizes):
            model = self._build_dense_layer(model, size, index)

        # Move Probability Output
        model_out = Dense(model_config.num_candidate_moves, kernel_regularizer=l2(model_config.l2_reg),
                          activation='softmax', name='output')(model)

        self.model = Model(input_layer, model_out, name='chess_move_model')

    def _build_dense_layer(self, model, size, index):
        model_config = self.config.move_probability_model
        layer_name = 'dense{}'.format(index)

        model = Dense(size, kernel_regularizer=l2(model_config.l2_reg), activation='sigmoid', name=layer_name)(model)

        return model

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import config


class ChessModel:
    """
    The convolutional residual deep neural network model that will be trained on board states to output
    value and policy predictions, like the AlphaZero implementation.
    """
    def __init__(self, cfg: config.Config):
        self.model = None
        self.config = cfg

    def build_model(self):
        """
        Build the Keras model
        """
        model_config = self.config.policy_model
        input_layer = model = Input(model_config.input_shape)

        # Build First convolution layer
        model = Conv2D(filters=model_config.cnn_filter_num,
                       kernel_size=model_config.cnn_first_filter_size,
                       padding='same',
                       data_format='channels_first',
                       use_bias=False,
                       kernel_regularizer=l2(model_config.l2_reg),
                       name='input_conv-{}-{}'.format(model_config.cnn_first_filter_size, model_config.cnn_filter_num)
                       )(model)

        model = BatchNormalization(axis=1, name='input_batchnorm')(model)
        model = Activation('relu', name='input_relu')(model)

        # Build residual blocks
        for i in range(model_config.res_layer_num):
            model = self._build_residual_block(model, i + 1)

        res_out = model

        # Policy Output
        policy_model = Conv2D(filters=2, kernel_size=1, data_format='channels_first', use_bias=False,
                              kernel_regularizer=l2(model_config.l2_reg), name='policy_conv-1-2')(res_out)
        policy_model = BatchNormalization(axis=1, name='policy_batchnorm')(policy_model)
        policy_model = Activation('relu', name='policy_relu')(policy_model)
        policy_model = Flatten(name='policy_flatten')(policy_model)
        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(model_config.l2_reg), activation='softmax',
                           name='policy_out')(policy_model)

        self.model = Model(input_layer, policy_out, name='chess_model')

    def _build_residual_block(self, model, index):
        model_config = self.config.policy_model
        residual_block_name = 'res{}'.format(index)
        input_model = model

        model = Conv2D(filters=model_config.cnn_filter_num,
                       kernel_size=model_config.cnn_filter_size,
                       padding='same',
                       data_format='channels_first',
                       use_bias=False,
                       kernel_regularizer=l2(model_config.l2_reg),
                       name='{}_conv1-{}-{}'.format(residual_block_name, model_config.cnn_first_filter_size, model_config.cnn_filter_num),
                       )(model)

        model = BatchNormalization(axis=1, name='{}_batchnorm1'.format(residual_block_name))(model)
        model = Activation('relu', name='{}_relu1'.format(residual_block_name))(model)

        model = Conv2D(filters=model_config.cnn_filter_num,
                       kernel_size=model_config.cnn_filter_size,
                       padding='same',
                       data_format='channels_first',
                       use_bias=False,
                       kernel_regularizer=l2(model_config.l2_reg),
                       name='{}_conv2-{}-{}'.format(residual_block_name, model_config.cnn_first_filter_size,
                                                    model_config.cnn_filter_num),
                       )(model)

        model = BatchNormalization(axis=1, name='{}_batchnorm2'.format(residual_block_name))(model)
        model = Add(name='{}_add'.format(residual_block_name))([input_model, model])
        model = Activation('relu', name='{}_relu2'.format(residual_block_name))(model)

        return model

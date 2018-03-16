from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from policy_dnn.model import ChessModel
from move_probability_nn.model import ChessMoveModel

import config
from move_probability_nn.data import ChessMoveDataGenerator


class ChessModelTrainer:
    def __init__(self, cfg: config.Config, model: ChessMoveModel, policy_model: ChessModel):
        self.config = cfg
        self.model = model
        self.policy_model = policy_model

    def train(self):
        compiled_model = self.model
        if not self.config.trainer.continue_from_best:
            self.compile_model()
            compiled_model = self.model.model

        train_from_file = self.config.trainer.train_type == self.config.training_type.FILE
        training_generator = ChessMoveDataGenerator(self.config, self.policy_model, from_file=train_from_file)\
            .generate()
        cross_validation_generator = ChessMoveDataGenerator(self.config, self.policy_model, from_file=train_from_file)\
            .generate()

        checkpoint_cb = ModelCheckpoint(filepath=self.config.resources.best_move_model_path,
                                        save_best_only=True,
                                        mode='min',
                                        monitor='val_loss')

        compiled_model.fit_generator(generator=training_generator,
                                     epochs=self.config.trainer.epochs,
                                     steps_per_epoch=self.config.trainer.steps_per_epoch,
                                     validation_steps=self.config.trainer.steps_per_epoch_cv,
                                     validation_data=cross_validation_generator,
                                     callbacks=[checkpoint_cb],
                                     verbose=1)

    def compile_model(self):
        optimizer = Adam()
        loss_function = 'categorical_crossentropy'
        self.model.model.compile(optimizer=optimizer, loss=loss_function)


if __name__ == '__main__':
    cfg = config.Config()
    policy_model = load_model(cfg.resources.best_model_path)

    if cfg.trainer.continue_from_best:
        model = load_model(cfg.resources.best_move_model_path)
    else:
        model = ChessMoveModel(cfg)
        model.build_model()

    trainer = ChessModelTrainer(cfg, model, policy_model)
    trainer.train()

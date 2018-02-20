from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import config

from model import ChessModel
from data_generator import ChessDataGenerator


class ChessModelTrainer:
    def __init__(self, cfg: config.Config, model: ChessModel):
        self.config = cfg
        self.model = model

    def train(self):
        compiled_model = self.model
        if not self.config.trainer.continue_from_best:
            self.compile_model()
            compiled_model = self.model.model

        training_generator = ChessDataGenerator(self.config, is_cross_validation=False).generate()
        cross_validation_generator = ChessDataGenerator(self.config, is_cross_validation=True).generate()

        checkpoint_cb = ModelCheckpoint(filepath=self.config.resources.best_model_path,
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
        loss_functions = ['categorical_crossentropy', 'mean_squared_error']
        loss_weights = self.config.trainer.loss_weights

        self.model.model.compile(optimizer=optimizer, loss=loss_functions, loss_weights=loss_weights)


if __name__ == '__main__':
    cfg = config.Config()

    if cfg.trainer.continue_from_best:
        model = load_model(cfg.resources.best_model_path)
    else:
        model = ChessModel(cfg)
        model.build_model()

    trainer = ChessModelTrainer(cfg, model)
    trainer.train()

"""
Generate data as a CSV file to train models
"""
import sys
import os
from keras.models import load_model

import config
from move_probability_nn.data import ChessMoveDataGenerator


def main(is_cv=False):
    cfg = config.Config()
    policy_model = load_model(cfg.resources.best_model_path)

    gen = ChessMoveDataGenerator(cfg, policy_model, is_cross_validation=is_cv, yield_meta=True)

    count = 0
    for df in gen.generate_data_frame():
        file_name = str(count) + ('_cv' if is_cv else '_tr') + '.csv'
        path = os.path.join(cfg.resources.move_data_directory, file_name)
        df.to_csv(path, index=False)

        count += 1
        print(count)

        if count > 1000:
            break


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(is_cv=True)

    main(is_cv=False)

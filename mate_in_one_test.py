from keras.models import load_model

import chess.pgn

import config
import game_parser


def main():
    cfg = config.Config()
    model = load_model(cfg.resources.best_model_path)

    pgn = open('/Users/tom/Projects/Portfolio/data/chess-deep-learning/test.pgn')
    game = chess.pgn.read_game(pgn)

    gp = game_parser.ChessGameParser(game, cfg)
    board_states, move_label_one_hots, value_labels = gp.get_training_examples()

    mate_in_one = board_states[-1].reshape((1, 20, 8, 8))

    policy, value = model.predict(mate_in_one)

    print(value)


if __name__ == '__main__':
    main()

"""
Simple CLI to play against the engine
"""
import chess
from keras.models import load_model

from chess_environment.position_parser import ChessPositionParser
from chess_environment.engine import Stockfish
from agent import mcts
import config


def main(engine_color):
    # Initialize the configuration
    cfg = config.Config()

    # Load up the trained model
    model = load_model(cfg.resources.best_model_path)
    num_simulations = 5000

    elo = 3000
    pp = ChessPositionParser(cfg, [chess.STARTING_FEN], [elo], fens_have_counters=True, elos_are_normalized=False)
    sf = Stockfish(config.Config())
    ts = mcts.ChessMonteCarloTreeSearch(cfg, model, pp, stockfish=sf, deterministic=True)

    board = chess.Board()
    while True:
        if board.turn == engine_color:
            print('Calculating next move...')
            next_move_uci = ts.set_position(board.fen(), num_simulations=num_simulations, tau=None, ucts_const=0.5).get_next_move()
            next_move = chess.Move.from_uci(next_move_uci)
            print(board.san(next_move))
            board.push(next_move)

        else:
            next_move_san = input('Next Move (san): ')
            if next_move_san == 'board':
                print(board.fen())
            else:
                try:
                    board.push_san(next_move_san)
                except ValueError:
                    print('Illegal Move. Try again.')


if __name__ == '__main__':
    main(chess.BLACK)

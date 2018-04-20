"""
Predicts probability distribution of the next move
"""
import numpy as np
import chess
from random import shuffle
from keras.models import load_model

from chess_environment.engine import Stockfish
from chess_environment.position_parser import ChessPositionParser
import config


class MovePredictor:
    def __init__(self, cfg: config.Config, policy_model=None, move_probability_model=None, stockfish=None,
                 position_parser=None):
        self.config = cfg

        self.policy_model = policy_model or load_model(self.config.resources.best_model_path)
        self.move_probability_model = move_probability_model or load_model(self.config.resources.best_move_model_path)
        self.stockfish = stockfish or Stockfish(self.config)
        self.position_parser = position_parser or ChessPositionParser(self.config, [], [])

        self.move_map = {chess.Move.from_uci(move): idx for idx, move in enumerate(self.config.labels)}

        self.num_candidate_moves = self.config.move_probability_model.num_candidate_moves
        self.num_evals = len(self.config.move_probability_model.valuation_depths)

    def predict(self, fen, elo, fen_has_counters=False, elo_is_normalized=True):
        board = chess.Board(self._full_fen(fen))
        is_white = board.turn == chess.WHITE
        if not elo_is_normalized:
            elo = self.normalize_elo(elo)

        policy_input = self.position_parser.reset(fens=[fen], elos=[elo], fens_have_counters=fen_has_counters,
                                                  elos_are_normalized=elo_is_normalized).get_canonical_input()

        policy = self.policy_model.predict(policy_input)[0]

        legal_moves = []
        for m in board.legal_moves:
            board.push(m)
            move = {
                'move': m,
                'policy': policy[self.move_map[m]],
                'valuations': []
            }
            for depth in self.config.move_probability_model.valuation_depths:
                move['valuations'].append(self.stockfish.eval(board, depth=depth, as_value=True))

            board.pop()
            legal_moves.append(move)

        legal_moves = sorted(legal_moves, key=lambda m: m['valuations'][-1], reverse=is_white)
        legal_moves = legal_moves[:self.num_candidate_moves]
        shuffle(legal_moves)

        evals = [-1.0] * (self.num_candidate_moves * self.num_evals)
        priors = [0.0] * self.num_candidate_moves
        for move_idx, move in enumerate(legal_moves):
            priors[move_idx] = move['policy']

            for val_idx, val in enumerate(move['valuations']):
                evals[move_idx * self.num_evals + val_idx] = val * (1 if is_white else -1)

        model_input = np.array([priors + evals + [float(elo)]])
        probs = self.move_probability_model.predict(model_input)[0]

        probs = probs[:len(legal_moves)]
        probs = (probs / probs.sum()).tolist()

        moves_out = {}
        for move, move_prob in zip(legal_moves, probs):
            moves_out[move['move']] = {
                'probability': move_prob,
                'value': move['valuations'][-1]
            }

        return moves_out

    @staticmethod
    def _full_fen(fen):
        if len(fen.split(' ')) == 6:
            return fen

        return fen + ' 0 1'

    @staticmethod
    def normalize_elo(elo):
        """
        Normalize elo on the range of -1 to 1, assuming min elo is 0 and max elo is 3000
        """
        return elo / 1500 - 1


if __name__ == '__main__':
    cfg = config.Config()
    mp = MovePredictor(cfg)

    fen = 'r1b1kbnr/pppp1ppp/2n5/4P3/1q3B2/5N2/PPP1PPPP/RN1QKB1R w KQkq -'
    elo = 0.0

    print(mp.predict(fen, elo))


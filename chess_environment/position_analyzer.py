import chess
import pandas as pd

from chess_environment.position_parser import ChessPositionParser
from chess_environment.engine import Stockfish
from agent import mcts
from util.data_migration_util import time_control_elo_adjustment

import config


class ChessPositionAnalyzer:
    def __init__(self, model, labels, fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                 elo=1500, time_control='3600+30'):

        self.config = config.Config()

        self.model = model
        self.labels = labels

        self.fen = fen
        self.elo = time_control_elo_adjustment(time_control) + elo
        self.time_control = time_control

        self.position_parser = ChessPositionParser(self.config, [fen], [elo],
                                                   fens_have_counters=True, elos_are_normalized=False)
        self.stockfish = Stockfish(config.Config())
        self.mcts = mcts.ChessMonteCarloTreeSearch(self.config, self.model, self.position_parser,
                                                   stockfish=self.stockfish)

    def analyze(self, mask_illegal=True, move_as_san=True,
                mcts_num_simulations=None, mcts_tau=None, mcts_ucts_const=None):

        if mcts_num_simulations:
            # Run monte carlo tree search and use the resulting value rollups and policy
            policy = self.mcts\
                .set_position(self.fen, num_simulations=mcts_num_simulations, tau=mcts_tau, ucts_const=mcts_ucts_const)\
                .get_mcts_policy()

        else:
            # Use the dnn policy without search
            dnn_input = self.position_parser.reset(
                fens=[self.fen], elos=[self.elo], fens_have_counters=True, elos_are_normalized=False).get_canonical_input()
            policy = self.model.predict(dnn_input)
            policy = policy[0].tolist()

        # Use the naked stockfish value without any depth
        value = self.stockfish.eval(chess.Board(self.fen))

        df = pd.DataFrame(list(zip(self.labels, policy)), columns=['Move', 'Probability'])

        if mask_illegal:
            board = self.board()
            legal_moves = [m.uci() for m in board.legal_moves]

            df = df[df['Move'].isin(legal_moves)]
            df['Probability'] = df['Probability'] / df['Probability'].sum()
            df = df.sort_values(by='Probability', ascending=False)

        if move_as_san:
            board = self.board()
            df['Move'] = df['Move'].apply(lambda uci: board.san(chess.Move.from_uci(uci)))

        return df, value

    def get_mcts_json(self):
        if self.mcts.move_probabilities is not None:
            return self.mcts.get_json()
        return '{}'

    def board(self):
        return chess.Board(self.fen)

    def reset(self, fen=None, elo=None, time_control=None):
        if fen is not None:
            self.fen = fen

        if time_control is not None:
            self.time_control = time_control

        if elo is not None:
            self.elo = time_control_elo_adjustment(self.time_control) + elo

        return self

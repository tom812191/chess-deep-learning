import chess
import pandas as pd

from game_parser import ChessPositionParser, ChessGameParser


class ChessPositionAnalyzer:
    def __init__(self, position_parser: ChessPositionParser, model, labels,
                 fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', white_elo=1500, black_elo=1500,
                 time_control='3600+30'):

        self.position_parser = position_parser

        self.model = model
        self.labels = labels

        self.fen = fen
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.time_control = time_control

    def analyze(self, mask_illegal=True):
        policy, value = self.model.predict(self.position_parser.reset(
            self.fen, self.white_elo, self.black_elo, self.time_control).input_tensor)

        value = value[0, 0]
        policy = policy[0].tolist()

        df = pd.DataFrame(list(zip(self.labels, policy)), columns=['Move', 'Probability'])

        if mask_illegal:
            board = self.board()
            legal_moves = [m.uci() for m in board.legal_moves]

            df = df[df['Move'].isin(legal_moves)]
            df['Probability'] = df['Probability'] / df['Probability'].sum()

        return df, value

    def board(self):
        return chess.Board(self.fen)

    def reset(self, fen=None, white_elo=None, black_elo=None, time_control=None):
        if fen is not None:
            self.fen = fen

        if white_elo is not None:
            self.white_elo = white_elo

        if black_elo is not None:
            self.black_elo = black_elo

        if time_control is not None:
            self.time_control = time_control

        return self

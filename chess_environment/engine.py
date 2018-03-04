import chess.uci

import config


class Stockfish:
    def __init__(self, cfg: config.Config):
        self.config = cfg

        self.stockfish = chess.uci.popen_engine(self.config.resources.stockfish_path)
        self.stockfish.uci()
        self.stockfish_info = chess.uci.InfoHandler()
        self.stockfish.info_handlers.append(self.stockfish_info)

    def eval(self, board):
        self.stockfish.position(board)
        self.stockfish.go(depth=1)
        cp = self.stockfish_info.info['score'][1].cp
        if cp is not None:
            return cp / 100.0

        # Mate in 1
        return 100.0

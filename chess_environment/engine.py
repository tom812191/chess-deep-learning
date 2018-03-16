import chess.uci

import config


class Stockfish:
    def __init__(self, cfg: config.Config, search_depth=5):
        self.config = cfg

        self.stockfish = chess.uci.popen_engine(self.config.resources.stockfish_path)
        self.stockfish.uci()
        self.stockfish_info = chess.uci.InfoHandler()
        self.stockfish.info_handlers.append(self.stockfish_info)
        self.search_depth = search_depth

    def eval(self, board, depth=None):
        # Check if game over
        if board.result() != '*':
            if board.result == '1/2-1/2':
                return 0.0

            # If the game is over and not a draw, then the player to move has lost (tree search doesn't resign)
            return -100.0

        d = self.search_depth
        if depth is not None:
            d = depth

        self.stockfish.position(board)
        self.stockfish.go(depth=d)

        score = self.stockfish_info.info['score'][1]
        if score.cp is not None:
            return self.stockfish_info.info["score"][1].cp / 100.0

        if score.mate > 0:
            # Giving mate
            return 100.0

        # Getting mated
        return -100.0

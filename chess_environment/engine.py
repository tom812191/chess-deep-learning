import config
import chess
import subprocess
import re


class Stockfish:
    def __init__(self, cfg: config.Config):
        self.config = cfg

        self.stockfish = subprocess.Popen(
            cfg.resources.stockfish_path,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

    def _send_command(self, command: str):
        self.stockfish.stdin.write(command + '\n')
        self.stockfish.stdin.flush()

    def _set_position(self, fen):
        self._send_command(f'position fen {fen}')

    def eval(self, board):
        """
        Static evaluation for player to move
        """
        # Check if game over
        if board.result() != '*':
            if board.result == '1/2-1/2':
                return int(0)

            # If the game is over and not a draw, then the player to move has lost (tree search doesn't resign)
            return int(-1)

        # Get Evaluation
        self._set_position(board.fen())
        self._send_command('eval')

        evaluation = re.compile('Total Evaluation: (-?\d+\.\d+) .*')

        while True:
            line = self.stockfish.stdout.readline()
            match = evaluation.match(line)
            if match:
                score = float(match.group(1))
                if board.turn == chess.BLACK:
                    score *= -1
                return float(score)

    def __del__(self):
        self.stockfish.kill()

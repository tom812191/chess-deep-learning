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
                return score

    def __del__(self):
        self.stockfish.kill()

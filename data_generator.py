"""
Create a data generator that will stream data on the fly from lichess.
"""
import numpy as np
import bz2
import requests

import config
import game_parser
import threading_util


class ChessDataGenerator:
    def __init__(self, cfg: config.Config, is_cross_validation=False):
        self.config = cfg
        self.batch_size = self.config.trainer.batch_size
        self.is_cross_validation = is_cross_validation

        self.download_url_list = self.get_download_url_list()
        self.download_chunk_size = self.config.resources.download_chunk_size
        self.yield_count = 0

    @threading_util.thread_safe_generator
    def generate(self):
        for pgn in self.iter_pgn_chunk():
            games = game_parser.ChessPGNParser(pgn, self.config).games
            board_states, move_label_one_hots, value_labels, state_meta_data = games

            indexes = self.get_exploration_order(games)
            board_states, move_label_one_hots, value_labels = (
                np.array(board_states),
                np.array(move_label_one_hots),
                np.array(value_labels),
            )

            # Generate batches
            i_max = int(len(indexes) / self.batch_size)
            for i in range(i_max):
                current_indexes = indexes[i * self.batch_size:(i + 1) * self.batch_size]
                X = board_states[current_indexes]
                y = [
                    move_label_one_hots[current_indexes],
                    value_labels[current_indexes],
                ]

                yield X, y
                self.yield_count += 1

    @staticmethod
    def get_exploration_order(games):
        """
        Take the games data and return a random exploration order
        """
        board_states, move_label_one_hots, value_labels, state_meta_data = games

        indexes = np.arange(len(board_states))
        np.random.shuffle(indexes)
        return indexes

    def iter_pgn_chunk(self):
        """
        A generator for downloading and parsing a chunk of data, specified by self.download_chunk_size
        :yields: a PGN string of games
        """
        # Initialize the decompressor
        decompressor = bz2.BZ2Decompressor()

        # Initialize variable to store partial game information (when chunk doesn't end at the end of a game)
        unused_data = ''

        # Iterate through download urls
        for url in self.iter_download_urls():

            # Iterate through file chunks
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=self.download_chunk_size):
                if chunk:

                    # Decompress chunk
                    try:
                        data = decompressor.decompress(chunk)
                    except OSError:
                        continue

                    if data == b'':
                        continue

                    # Decode chunk
                    pgn = unused_data + data.decode('utf-8')
                    pgn, unused_data = ChessDataGenerator.split_partial_game(pgn)

                    yield pgn

    def get_download_url_list(self):
        list_path = self.config.resources.lichess_download_list
        if self.is_cross_validation:
            list_path = self.config.resources.lichess_download_list_cv

        with open(list_path) as f:
            return [line.strip() for line in f.readlines()]

    @threading_util.thread_safe_generator
    def iter_download_urls(self):
        idx = 0
        while True:
            yield self.download_url_list[idx]
            idx = (idx + 1) % len(self.download_url_list)

    @staticmethod
    def split_partial_game(pgn: str):
        """
        Take a string of PGN games from a file stream and split into all the complete games and the
        final piece of a game
        :return: games, partial_game
        """
        idx = pgn.rfind('[Event')
        return pgn[:idx], pgn[idx:]

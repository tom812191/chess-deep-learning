"""
Create a data generator that will stream data on the fly from lichess.
"""
import numpy as np

import bz2
import requests
import os

import json

import config
from chess_environment import game_parser
from util import threading_util
from util.data_migration_util import connect


class ChessFileDataGenerator:
    """
    Data generator for training of the neural network using data from a tsv file
    """
    def __init__(self, cfg: config.Config, is_cross_validation=False):
        self.config = cfg
        self.is_cross_validation = is_cross_validation

        self.batch_size = self.config.trainer.batch_size

    @threading_util.thread_safe_generator
    def generate(self):

        file_path = os.path.join(
            self.config.resources.data_directory,
            self.config.trainer.cv_file_name if self.is_cross_validation else self.config.trainer.train_file_name
        )

        row_batch_size = 100

        while True:
            with open(file_path) as f:
                current_rows = []
                for row in f:
                    current_row = row.split('\t')
                    current_row[-1] = current_row[-1].replace('\\n', '\n')
                    current_rows.append(current_row)
                    if len(current_rows) >= row_batch_size:
                        games = game_parser.ChessDatabaseParser(current_rows, self.config).games

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

                        current_rows = []

    @staticmethod
    def get_exploration_order(games):
        """
        Take the games data and return a random exploration order
        """
        board_states, move_label_one_hots, value_labels, state_meta_data = games

        indexes = np.arange(len(board_states))
        np.random.shuffle(indexes)
        return indexes



class ChessCurriculumDataGenerator:
    """
    Data generator for training of the neural network using strategic ordering of the data.
    """
    def __init__(self, cfg: config.Config, is_cross_validation=False):
        self.config = cfg
        self.batch_size = self.config.trainer.batch_size
        self.is_cross_validation = is_cross_validation

        self.db_conn = self.db_connect()

    @threading_util.thread_safe_generator
    def generate(self):
        cursor = self.db_conn.cursor()
        query = """
            
            SELECT
                white_elo,
                black_elo,
                time_control,
                pgn
            FROM
                games
            WHERE
                partition = '{partition}' AND {conditions}
        """

        # Infinite generator loop
        while True:

            # Loop through curriculum conditions
            for curriculum in self.config.trainer.curriculum:
                query_params = {
                    'partition': 'cv' if self.is_cross_validation else 'tr',
                    'conditions': ' AND '.join(curriculum['query_conditions'])
                }

                q = query.format(**query_params)
                cursor.execute(q)
                self.db_conn.commit()

                # Loop through rows
                done = False
                games_per_fetch = 100
                while not done:
                    rows = cursor.fetchmany(games_per_fetch)
                    games = game_parser.ChessDatabaseParser(rows, self.config).games

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

                    if len(rows) < games_per_fetch:
                        done = True

    @staticmethod
    def get_exploration_order(games):
        """
        Take the games data and return a random exploration order
        """
        board_states, move_label_one_hots, value_labels, state_meta_data = games

        indexes = np.arange(len(board_states))
        np.random.shuffle(indexes)
        return indexes

    def db_connect(self):
        with open(self.config.resources.database_settings_file) as f:
            db_settings = json.load(f)

        return connect(db_settings)


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

        error_count = 0
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
                        error_count += 1
                        print(f'Error decompressing chunk: #{error_count}')
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
        indexes = np.arange(len(self.download_url_list))
        np.random.shuffle(indexes)
        while True:
            yield self.download_url_list[indexes[idx]]
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

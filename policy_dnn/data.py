"""
Data generator for training
"""
import numpy as np
import json
import pymongo
import os
import bz2

import config
from chess_environment import position_parser
from util import threading_util


class ChessDataGenerator:
    def __init__(self, cfg: config.Config, from_file=False, is_cross_validation=False):
        self.config = cfg
        self.batch_size = self.config.trainer.batch_size
        self.is_cross_validation= is_cross_validation

        self.from_file = from_file

        with open(cfg.resources.database_settings_file) as f:
            self.db_settings = json.load(f)

        self.elo_map = {
            'lt_1500': position_parser.ChessPositionParser.normalize_elo(1250),
            '1500_2000': position_parser.ChessPositionParser.normalize_elo(1750),
            '2000_2500': position_parser.ChessPositionParser.normalize_elo(2250),
            'gt_2500': position_parser.ChessPositionParser.normalize_elo(2750),
        }

        self.move_labels = self.config.labels
        self.num_moves = len(self.move_labels)
        self.move_indexes = {move: idx for idx, move in enumerate(self.move_labels)}

    def generate(self):
        if self.from_file:
            return self.generate_from_file()
        return self.generate_from_db()

    @threading_util.thread_safe_generator
    def generate_from_db(self):
        client = pymongo.MongoClient(self.db_settings['host'], self.db_settings['port'])
        db = client.chess

        while True:
            # Get positions that have been played at least min_position_visits_total times
            cursor = db.positions.find(
                {
                    'total': {
                        '$gte': self.config.trainer.min_position_visits_total,
                    }
                }, no_cursor_timeout=True
            )

            state, move_frequency = [], []
            for doc in cursor:

                # Hack to split into training and cross validation
                doc_is_cv = not bool(doc['total'] % 10)
                if doc_is_cv != self.is_cross_validation:
                    continue

                doc_states, doc_move_frequencies = self.process_doc(doc)
                state += doc_states
                move_frequency += doc_move_frequencies

                if len(state) >= self.batch_size:
                    # np.array(batch[:self.batch_size])
                    fens = [s['fen'] for s in state[:self.batch_size]]
                    elos = [s['elo'] for s in state[:self.batch_size]]
                    X = position_parser.ChessPositionParser(self.config, fens, elos, fens_have_counters=False)\
                        .get_canonical_input()
                    y = np.array(move_frequency[:self.batch_size])
                    yield X, y
                    state = state[self.batch_size:]
                    move_frequency = move_frequency[self.batch_size:]

            cursor.close()

    @threading_util.thread_safe_generator
    def generate_from_file(self):
        dir = self.config.resources.training_data_directory
        cv_str = 'cv_' if self.is_cross_validation else ''

        while True:
            file_count = 1
            while os.path.exists(os.path.join(dir, f'X_{cv_str}{file_count}.npy.bz2')):
                batch_count = 0

                ChessDataGenerator.decompress_file(os.path.join(dir, f'X_{cv_str}{file_count}.npy.bz2'),
                                                   os.path.join(dir, f'X_{cv_str}{file_count}.npy'))

                ChessDataGenerator.decompress_file(os.path.join(dir, f'y_{cv_str}{file_count}.npy.bz2'),
                                                   os.path.join(dir, f'y_{cv_str}{file_count}.npy'))

                X_batches = np.load(os.path.join(dir, f'X_{cv_str}{file_count}.npy'))
                y_batches = np.load(os.path.join(dir, f'y_{cv_str}{file_count}.npy'))
                finished_file = False
                while not finished_file:
                    indexes = slice(batch_count * self.batch_size, (batch_count+1) * self.batch_size)
                    X = X_batches[indexes]
                    y = y_batches[indexes]

                    if len(X > 0) and len(y > 0):
                        yield X, y

                    batch_count += 1
                    if len(X) < self.batch_size:
                        finished_file = True

                os.remove(os.path.join(dir, f'X_{cv_str}{file_count}.npy'))
                os.remove(os.path.join(dir, f'y_{cv_str}{file_count}.npy'))
                file_count += 1

    def process_doc(self, doc):
        fen = doc['fen']
        state, move_frequency = [], []
        for elo_range, elo_value in self.elo_map.items():
            if elo_range in doc:
                moves = np.zeros((self.num_moves,))
                for move, move_count in doc[elo_range].items():
                    moves[self.move_indexes[move]] = move_count
                total_moves = moves.sum()
                if total_moves >= self.config.trainer.min_position_visits:
                    move_frequency.append(moves/total_moves)
                    state.append({'fen': fen, 'elo': elo_value})

        return state, move_frequency

    @staticmethod
    def decompress_file(compressed_path, decompressed_path):
        with open(decompressed_path, 'wb') as new_file, bz2.BZ2File(compressed_path, 'rb') as f:
            for data in iter(lambda: f.read(100 * 1024), b''):
                new_file.write(data)


def generate_to_file(is_cross_validation=False):
    cfg = config.Config()
    dir = cfg.resources.training_data_directory
    cv_str = 'cv_' if is_cross_validation else ''

    cdg = ChessDataGenerator(cfg, is_cross_validation=is_cross_validation)

    total_batches = cfg.trainer.steps_per_epoch_cv if is_cross_validation else cfg.trainer.steps_per_epoch
    batches_per_file = 100
    batch_count = 1
    file_count = 1

    X_batches = None
    y_batches = None

    for X, y in cdg.generate():
        if batch_count > total_batches:
            break

        if X_batches is None:
            X_batches = X
            y_batches = y
        else:
            X_batches = np.concatenate((X_batches, X))
            y_batches = np.concatenate((y_batches, y))

        if batch_count % batches_per_file == 0:
            np.save(os.path.join(dir, f'X_{cv_str}{file_count}.npy'), X_batches)
            np.save(os.path.join(dir, f'y_{cv_str}{file_count}.npy'), y_batches)
            file_count += 1
            X_batches = None
            y_batches = None

        batch_count += 1

    np.save(os.path.join(dir, f'X_{cv_str}{file_count}.npy'), X_batches)
    np.save(os.path.join(dir, f'y_{cv_str}{file_count}.npy'), y_batches)

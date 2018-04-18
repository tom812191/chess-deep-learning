"""
Data generator for training
"""
import numpy as np
import pandas as pd
import json
import pymongo
import chess
from itertools import product

import config
from chess_environment import position_parser, engine
from util import threading_util


class ChessMoveDataGenerator:
    def __init__(self, cfg: config.Config, policy_model, from_file=False, is_cross_validation=False,
                 yield_meta=False):
        self.config = cfg
        self.batch_size = self.config.trainer.batch_size
        self.is_cross_validation = is_cross_validation
        self.yield_meta = yield_meta

        self.from_file = from_file

        with open(cfg.resources.database_settings_file) as f:
            self.db_settings = json.load(f)

        self.elo_map = {
            'lt_1500': position_parser.ChessPositionParser.normalize_elo(1250),
            '1500_2000': position_parser.ChessPositionParser.normalize_elo(1750),
            '2000_2500': position_parser.ChessPositionParser.normalize_elo(2250),
            'gt_2500': position_parser.ChessPositionParser.normalize_elo(2750),
        }

        self.policy_model = policy_model
        self.position_parser = position_parser.ChessPositionParser(self.config, [], [])
        self.board = chess.Board()
        self.stockfish = engine.Stockfish(self.config)
        self.move_map = {chess.Move.from_uci(move): idx for idx, move in enumerate(self.config.labels)}

        self.num_candidate_moves = self.config.move_probability_model.num_candidate_moves
        self.num_evals = len(self.config.move_probability_model.valuation_depths)

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

            fens, elos, moves = [], [], []
            doc_count = 0
            for doc in cursor:

                # Hack to split into training and cross validation
                doc_is_cv = not bool(doc['total'] % 10)
                if doc_is_cv != self.is_cross_validation:
                    continue

                doc_count += 1

                doc_fens, doc_elos, doc_moves = self.process_doc(doc)
                fens += doc_fens
                elos += doc_elos
                moves += doc_moves

                if len(fens) >= self.batch_size:
                    fens_batch = fens[:self.batch_size]
                    elos_batch = elos[:self.batch_size]
                    moves_batch = moves[:self.batch_size]

                    if self.yield_meta:
                        X, y, fens, moves = self.process_batch(fens_batch, elos_batch, moves_batch)
                        yield X, y, fens, moves
                    else:
                        X, y = self.process_batch(fens_batch, elos_batch, moves_batch)
                        yield X, y

                    fens = fens[self.batch_size:]
                    elos = elos[self.batch_size:]
                    moves = moves[self.batch_size:]

            cursor.close()

    @threading_util.thread_safe_generator
    def generate_from_file(self):
        raise NotImplementedError('From file generator not implemented')

    def process_doc(self, doc):
        fen = doc['fen']
        board = self.board.copy()
        board.set_fen(fen + ' 0 1')

        # Parse moves from doc
        elos = []
        fens = []
        moves = []
        for elo_range, elo_value in self.elo_map.items():
            if elo_range in doc:
                total = sum([move_count for _, move_count in doc[elo_range].items()])

                if total >= self.config.trainer.min_position_visits:
                    elos.append(elo_value)
                    fens.append(fen)
                    moves.append([{
                        'uci': m.uci(),
                        'count': doc[elo_range][m.uci()] if m.uci() in doc[elo_range] else 0,
                    } for m in board.legal_moves])

        moves = [sorted(m, key=lambda x: x['count'], reverse=True) for m in moves]

        return fens, elos, moves

    def process_batch(self, fens, elos, moves):
        policies = self.policy_model.predict(self.position_parser.reset(fens=fens, elos=elos).get_canonical_input())

        evals = []
        priors = []
        counts = []

        for fen, policy, position_moves in zip(fens, policies.tolist(), moves):
            board = self.board.copy()
            board.set_fen(fen + ' 0 1')
            is_white = board.turn == chess.WHITE

            position_evals = [-1] * (self.num_candidate_moves * self.num_evals)
            position_priors = [0] * self.num_candidate_moves
            position_counts = [0] * self.num_candidate_moves
            for move_idx, move in enumerate(position_moves):
                if move_idx >= self.num_candidate_moves:
                    break

                m = chess.Move.from_uci(move['uci'])

                position_priors[move_idx] = policy[self.move_map[m]]
                position_counts[move_idx] = move['count']

                board.push(m)
                for val_idx, depth in enumerate(self.config.move_probability_model.valuation_depths):
                    value = self.stockfish.eval(self.board, depth=depth, as_value=True)

                    if not is_white:
                        value *= -1

                    position_evals[move_idx * self.num_evals + val_idx] = value
                board.pop()

            evals.append(position_evals)
            priors.append(position_priors)
            counts.append(position_counts)

        evals = np.array(evals)
        elos = np.array(elos).reshape((-1, 1,))
        priors = np.array(priors)
        counts = np.array(counts)

        priors = priors / priors.sum(axis=1)[:, np.newaxis]

        X = np.concatenate((priors, evals, elos), axis=1)
        y = counts / counts.sum(axis=1)[:, np.newaxis]

        assert len(X) == len(y)
        assert y.shape[1] == self.num_candidate_moves
        assert X.shape[1] == self.num_candidate_moves * (self.num_evals + 1) + 1

        if self.yield_meta:
            return X, y, fens, moves

        return X, y

    def generate_data_frame(self):
        assert self.yield_meta

        for X, y, fens, moves in self.generate():
            columns_X = [f'prior_{m}' for m in range(self.num_candidate_moves)] + \
                        [f'eval_{m}_d{d}' for m, d in product(range(self.num_candidate_moves), self.config.move_probability_model.valuation_depths)] + \
                        ['elo']
            df_X = pd.DataFrame(X, columns=columns_X)

            columns_y = [f'actual_freq_{m}' for m in range(self.num_candidate_moves)]
            df_y = pd.DataFrame(y, columns=columns_y)

            df_fens = pd.DataFrame(fens, columns=['fen'])

            columns_moves = [f'move_{m}' for m in range(self.num_candidate_moves)]
            data_moves = [[m['uci'] for m in move] for move in moves]
            df_moves = pd.DataFrame(data_moves, columns=columns_moves)

            yield pd.concat((df_fens, df_moves, df_X, df_y), axis=1)




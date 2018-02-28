"""
Functions to pull data from Lichess and populate a database
"""
import json
import sys

import bz2
from io import StringIO
import pymongo
from collections import defaultdict

import chess
import chess.pgn

import config


def main(load_lichess=False, load_millionbase=False):
    cfg = config.Config()

    with open(cfg.resources.database_settings_file) as f:
        db_settings = json.load(f)

    client = pymongo.MongoClient(db_settings['host'], db_settings['port'])
    chunk_size = 1024 * 1024 * 3  # 3 MB

    if load_lichess:

        lichess_files = [
            ('201712', '2017-12'),
        ]

        for file in lichess_files:
            file_id = file[0]
            file_path = '/Users/tom/Projects/Portfolio/data/chess-deep-learning/lichess_db_standard_rated_{}.pgn.bz2'.format(file[1])
            populate_db(client, file_path, chunk_size=chunk_size)
            print('Finished file {}'.format(file_id))

    if load_millionbase:
        millionbase_file_path = '/Users/tom/Projects/Portfolio/data/chess-deep-learning/millionbase-2.22.pgn.bz2'
        populate_db(client, millionbase_file_path, chunk_size=chunk_size)


def populate_db(client, file_path, chunk_size=1024):

    # Initialize the decompressor
    decompressor = bz2.BZ2Decompressor()

    # Initialize variable to store partial game information (when chunk doesn't end at the end of a game)
    unused_data = ''

    chunks_loaded = 0
    with open(file_path, 'rb') as f:
        print('Open file {}'.format(file_path))
        for compressed_data in iter(lambda: f.read(chunk_size), b''):
            data = decompressor.decompress(compressed_data)

            if data == b'':
                continue

            # Decode chunk
            pgn = unused_data + data.decode('utf-8')
            pgn, unused_data = split_partial_game(pgn)

            process_games(client, pgn)

            chunks_loaded += 1
            print('Processed {} MB'.format(int(chunks_loaded * chunk_size / (1024 * 1024))))


def process_games(client, pgn):
    pgn = StringIO(pgn)
    db = client.chess

    game = chess.pgn.read_game(pgn)
    positions = defaultdict(default_document)
    game_count = 0

    # Only look at positions up to 20 half moves in
    half_move_limit = 20

    # Cache time control elo adjustments to save time
    tc_cache = {}

    while game is not None:

        if game_count % 1000 == 0:
            sys.stdout.write('\rGame {}'.format(game_count))
            sys.stdout.flush()

        # Iterate over moves and store positions
        board = game.board()
        white_to_move = True

        white_elo = int(game.headers['WhiteElo']) if 'WhiteElo' in game.headers else 2300
        black_elo = int(game.headers['BlackElo']) if 'BlackElo' in game.headers else 2300

        # Adjust elo for time control
        tc = game.headers['TimeControl'] if 'TimeControl' in game.headers else '5400+30'
        elo_adj = tc_cache[tc] if tc in tc_cache else time_control_elo_adjustment(tc)

        white_elo = map_elo(white_elo + elo_adj)
        black_elo = map_elo(black_elo + elo_adj)

        half_move_counter = 0
        for move in game.main_line():
            half_move_counter += 1
            if half_move_counter > half_move_limit:
                break

            fen = ' '.join(board.fen().split(' ')[:-2])
            move_uci = move.uci()

            positions[fen][white_elo if white_to_move else black_elo][move_uci] += 1

            board.push(move)
            white_to_move = not white_to_move

        game = chess.pgn.read_game(pgn)
        game_count += 1

    bulk_operations = []
    for fen, elo_dict in positions.items():
        increment = {'total': 0}
        for elo, moves_dict in elo_dict.items():
            for move, count in moves_dict.items():
                increment[f'{elo}.{move}'] = count
                increment['total'] += count

        bulk_operations.append(pymongo.UpdateOne(
            {'fen': fen},

            {
                '$inc': increment,
            }, upsert=True
        ))

    print('\nStart DB Write')
    db.positions.bulk_write(bulk_operations)


def default_document():
    def default_move():
        return 0

    return {
        'lt_1500': defaultdict(default_move),
        '1500_2000': defaultdict(default_move),
        '2000_2500': defaultdict(default_move),
        'gt_2500': defaultdict(default_move),
    }


def map_elo(elo):
    if elo < 1500:
        return 'lt_1500'
    if elo < 2000:
        return '1500_2000'
    if elo < 2500:
        return '2000_2500'
    return 'gt_2500'


def split_partial_game(pgn: str):
    """
    Take a string of PGN games from a file stream and split into all the complete games and the
    final piece of a game
    :return: games, partial_game
    """
    idx = pgn.rfind('[Event')
    return pgn[:idx], pgn[idx:]


def time_control_elo_adjustment(time_control: str):
        """
        Adjust elo based on the time controls.

        Shorter time controls correspond to a worse elo. Adjust linearly with a max adjustment of -500
        """
        max_tc = 90 * 60 + 30 * 40
        if time_control == '-':
            seconds = max_tc
        else:
            tc = time_control.split('+')
            clock, increment = tc[0], 0

            if len(tc) > 1:
                increment = tc[1]

            seconds = min(int(clock) + int(increment) * 40, max_tc)

        a = 500 / (6600**2)

        return -a * (seconds - 6600)**2


if __name__ == '__main__':
    main(
        load_lichess=True,
        load_millionbase=True
    )


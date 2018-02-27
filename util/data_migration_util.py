"""
Functions to pull data from Lichess and populate a database
"""
import json
import sys
import re

import bz2
from io import StringIO
from util.db_util import connect, create_table

import config


def main(recreate_table=False, load_lichess=False, load_millionbase=False):
    cfg = config.Config()

    with open(cfg.resources.database_settings_file) as f:
        db_settings = json.load(f)

    conn = connect(db_settings)

    if recreate_table:
        table_name = 'games'
        table_cols = [
            'id serial PRIMARY KEY',
            'white VARCHAR(64)',
            'black VARCHAR(64)',
            'white_elo SMALLINT',
            'black_elo SMALLINT',
            'time_control VARCHAR(32)',
            'opening VARCHAR(128)',
            'eco CHAR(3)',
            'has_checkmate BOOL',
            'partition CHAR(2)',
            'file_id CHAR(6)',
            'game_number INTEGER',
            'pgn TEXT',
            'result SMALLINT',
        ]

        create_table(conn, table_name, table_cols, recreate=True)

    if load_lichess:

        lichess_files = [
            ('201801', '2018-01', 'tr'),
            ('201712', '2017-12', 'tr'),
            ('201711', '2017-11', 'tr'),
            ('201710', '2017-10', 'tr'),
            ('201709', '2017-09', 'tr'),
            ('201708', '2017-08', 'tr'),
            ('201707', '2017-07', 'cv'),
            ('201706', '2017-06', 'cv'),
        ]

        chunk_size = 1024 * 1024 * 8  # 8 MB
        for file in lichess_files:
            file_id = file[0]
            file_path = '/Users/tom/Projects/Portfolio/data/chess-deep-learning/lichess_db_standard_rated_{}.pgn.bz2'.format(file[1])
            partition = file[2]
            populate_db(conn, file_path, file_id, partition, chunk_size=chunk_size)
            print('Finished file {}'.format(file_id))

    if load_millionbase:
        millionbase_pgn_path = '/Users/tom/Projects/Portfolio/data/chess-deep-learning/millionbase-2.22.pgn'
        populate_games(conn, millionbase_pgn_path, 'tr', 'MLBASE', 0, pgn_is_file_path=True)


def populate_db(conn, file_path, file_id, partition, chunk_size=1024):

    # Initialize the decompressor
    decompressor = bz2.BZ2Decompressor()

    # Initialize variable to store partial game information (when chunk doesn't end at the end of a game)
    unused_data = ''

    games_loaded = 0
    chunks_loaded = 0
    with open(file_path, 'rb') as f:
        for compressed_data in iter(lambda: f.read(chunk_size), b''):
            data = decompressor.decompress(compressed_data)

            if data == b'':
                continue

            # Decode chunk
            pgn = unused_data + data.decode('utf-8')
            pgn, unused_data = split_partial_game(pgn)

            count = populate_games(conn, pgn, partition, file_id, games_loaded)
            games_loaded += count
            chunks_loaded += 1

            sys.stdout.write('\rProcessed {} games, {} MB'.format(games_loaded,
                                                                  int(chunks_loaded * chunk_size / (1024 * 1024))))
            sys.stdout.flush()


def populate_games(conn, pgn, partition, file_id, game_number_offset, pgn_is_file_path=False):
    cur = conn.cursor()
    game_count = 0
    result_map = {
        '1-0': 1,
        '0-1': -1,
        '1/2-1/2': 0,
    }

    if pgn_is_file_path:
        pgn_file = open(pgn)
    else:
        pgn_file = StringIO(pgn)

    for game in iter_pgn(pgn_file):
        try:
            data = {
                'white': game['headers']['White'],
                'black': game['headers']['Black'],
                'white_elo': game['headers']['WhiteElo'] if 'WhiteElo' in game['headers'] else 2300,
                'black_elo': game['headers']['BlackElo'] if 'BlackElo' in game['headers'] else 2300,
                'time_control': game['headers']['TimeControl'] if 'TimeControl' in game['headers'] else '6600+0',
                'opening': game['headers']['Opening'] if 'Opening' in game['headers'] else '',
                'eco': game['headers']['ECO'] if 'ECO' in game['headers'] else '',
                'has_checkmate': game['has_checkmate'],
                'partition': partition,
                'file_id': file_id,
                'game_number': game_count + game_number_offset,
                'pgn': game['pgn'],
                'result': result_map[game['headers']['Result']]
            }
        except KeyError:
            continue

        q = """
        INSERT INTO games (
            white, 
            black, 
            white_elo, 
            black_elo, 
            time_control, 
            opening, 
            eco, 
            has_checkmate, 
            partition, 
            file_id,
            game_number,
            pgn,
            result
        ) VALUES (
            %(white)s,
            %(black)s,
            %(white_elo)s,
            %(black_elo)s,
            %(time_control)s,
            %(opening)s,
            %(eco)s,
            %(has_checkmate)s,
            %(partition)s,
            %(file_id)s,
            %(game_number)s,
            %(pgn)s,
            %(result)s
        )
        """

        cur.execute(q, data)
        game_count += 1

    conn.commit()

    if pgn_is_file_path:
        pgn_file.close()

    return game_count


def split_partial_game(pgn: str):
    """
    Take a string of PGN games from a file stream and split into all the complete games and the
    final piece of a game
    :return: games, partial_game
    """
    idx = pgn.rfind('[Event')
    return pgn[:idx], pgn[idx:]


def iter_pgn(pgn_file):
    """
    Generator to iterate over a pgn file with multiple games
    """
    current_headers = {}
    header_re = re.compile('\[([a-zA-Z]+) "(.*)"\]')
    current_game = []

    for l in pgn_file:
        line = l.strip()

        # Check for new game
        if line.startswith('[Event'):
            current_game = []
            current_headers = {}

        current_game.append(l)
        match = header_re.match(line)
        if match:
            current_headers[match.group(1)] = match.group(2)

        # A non-blank line that doesn't match headers will be the game
        elif line != '':
            yield {
                'headers': current_headers,
                'pgn': ''.join(current_game),
                'has_checkmate': '#' in line,
            }


if __name__ == '__main__':
    main(
        recreate_table=False,
        load_lichess=False,
        load_millionbase=True
    )

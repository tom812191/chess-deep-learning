"""
Create a position table in the database to train the network on
"""
import json

from util.db_util import connect, create_table

import config


def main(recreate_table=False):
    cfg = config.Config()

    with open(cfg.resources.database_settings_file) as f:
        db_settings = json.load(f)

    conn = connect(db_settings)

    if recreate_table:
        table_name = 'positions'
        table_cols = [
            'fen VARCHAR(80)',
            'elo SMALLINT',
            'time_control VARCHAR(32)',
            'move_uci VARCHAR(5)',
        ]

        create_table(conn, table_name, table_cols, recreate=True)


def transfer_tables(conn, chunk_size=1000):
    """
    Take data out of the games table and put it into the positions table
    """
    # Get a chunk of games
    query = """
        SELECT
            white_elo,
            black_elo,
            time_control,
            pgn
        FROM
            games
        LIMIT
            %(limit)s
        OFFSET
            %(offset)s
    """


if __name__ == '__main__':
    main()

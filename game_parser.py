import numpy as np
from io import StringIO
import warnings

import chess.pgn

import config


class ChessPGNParser:
    """
    Utility to parse a PGN string into canonical state representations
    """
    def __init__(self, pgn: str, cfg: config.Config):
        self.pgn = pgn
        self.config = cfg

        self.board_states = None
        self.move_label_one_hots = None
        self.value_labels = None
        self.state_meta_data = None

    def parse_pgn(self):
        pgn = StringIO(self.pgn)

        self.board_states = []
        self.move_label_one_hots = []
        self.value_labels = []
        self.state_meta_data = []

        game = chess.pgn.read_game(pgn)
        while game is not None:
            try:
                parsed_game = ChessGameParser(game, self.config)
            except (KeyError, ValueError) as err:
                if self.config.show_warnings:
                    warnings.warn(err)
                game = chess.pgn.read_game(pgn)
                continue

            # Get canonical data representations for neural net, aka X, y_policy, y_value
            board_states, move_label_one_hots, value_labels = parsed_game.get_training_examples()

            # Get meta data. Will be used to order inputs for curriculum learning
            white_elo, black_elo = parsed_game.white_elo, parsed_game.black_elo
            has_checkmate = parsed_game.has_checkmate

            meta = []
            for idx in range(len(board_states)):
                meta.append({
                    'agent_elo': black_elo if idx % 2 else white_elo,
                    'opponent_elo': white_elo if idx % 2 else black_elo,
                    'has_checkmate': has_checkmate,
                    'next_halfmove_number': idx + 1,
                    'total_halfmoves': len(board_states),
                })

            self.board_states += board_states
            self.move_label_one_hots += move_label_one_hots
            self.value_labels += value_labels
            self.state_meta_data += meta

            game = chess.pgn.read_game(pgn)

    @property
    def games(self):
        if self.board_states is None:
            self.parse_pgn()
        return self.board_states, self.move_label_one_hots, self.value_labels, self.state_meta_data


class ChessGameParser:
    """
    Utility to convert games loaded from chess.pgn.read_game to training examples for the policy and value network.

    8x8 board with 20 channels
    The 20 channels are:
        12 for piece type: KQRBNPkqrbnp
         4 for castling KQkq
         1 for en passant
         2 for white elo, black, elo
         1 for time control
    """

    def __init__(self, game, cfg: config.Config):
        """
        Create the chess game
        :param game: Game loaded from PGN with chess.pgn.read_game
        :param cfg: Config class
        """
        self.config = cfg
        self.game = game
        self.white_elo = ChessGameParser.normalize_elo(int(game.headers['WhiteElo']))
        self.black_elo = ChessGameParser.normalize_elo(int(game.headers['BlackElo']))
        self.time_control = ChessGameParser.parse_time_control(game.headers['TimeControl'])
        self.result = ChessGameParser.parse_result(game.headers['Result'])

    @property
    def has_checkmate(self):
        """
        True if the game ends in checkmate, False otherwise
        """
        return self.game.end().board().is_checkmate()

    def get_training_examples(self):
        """
        Get training examples in canonical form, i.e. a list of np.arrays of shape (20, 8, 8).

        The 20 channels are:
            12 for piece type: KQRBNPkqrbnp
             4 for castling KQkq
             1 for en passant
             2 for agent elo, opponent, elo
             1 for time control

        For each board state, have it such that it is the agent's turn to move, and the board is represented from the
        agent's perspective. As far as the game FEN is concerned, if it is black to move, then flip piece color.

        :return: np.array
        """
        board = self.game.board()
        board_states = []
        value_labels = []
        move_label_one_hots = []

        move_labels = np.array(self.config.labels)

        # Create a board state after each move. Ignore the final state, since we don't have a next move to predict
        for move in self.game.main_line():
            fen = board.fen()

            board_states.append(self.get_canonical_input(fen))
            value_labels.append(-1 * self.result if ChessGameParser.black_to_move(fen) else self.result)

            move_uci = move.uci()
            move_label = move_labels == move_uci
            move_label_one_hots.append(move_label.astype(np.float32))

            board.push(move)

        return board_states, move_label_one_hots, value_labels

    def get_canonical_input(self, fen):
        """
        Create the canonical input for the chess state according to the inputs

        :param fen: State of the board in fen notation
        :return:
        """
        agent_elo = self.white_elo
        opponent_elo = self.black_elo

        if ChessGameParser.black_to_move(fen):
            fen = ChessGameParser.flip_fen(fen)
            agent_elo, opponent_elo = opponent_elo, agent_elo

        piece_state = self.fen_to_piece_state(fen)
        castling_state = self.fen_to_castling_state(fen)
        en_passant_state = self.fen_to_en_passant_state(fen)
        agent_elo_state = np.full((1, 8, 8), agent_elo)
        opponent_elo_state = np.full((1, 8, 8), opponent_elo)
        time_control_state = np.full((1, 8, 8), self.time_control)

        canonical_input = np.vstack((
            piece_state,
            castling_state,
            en_passant_state,
            agent_elo_state,
            opponent_elo_state,
            time_control_state,
        ))

        assert canonical_input.shape == (20, 8, 8)
        return canonical_input

    def fen_to_piece_state(self, fen: str):
        """
        Take the board state for the pieces as a FEN string and convert it to a 12x8x8 np.array, where the first
        dimension is the piece type in self.config.piece_order and values are 1 if and only if the piece type occupies
        the square.

        :param fen: The game state in fen format
        :return: The game state in np.array format
        """
        # Initialize empty board
        board_array = np.zeros(shape=(12, 8, 8), dtype=np.float32)

        # Replace fen numbers (indicating the number of blank squares) with the same number of zeros.
        # E.g. '3' becomes '000'.
        fen = ''.join(['0' * int(c) if c.isdigit() else c for c in fen])

        # Extract only the board position (i.e. ignore castling, move, en passant, etc.)
        fen = fen.split(' ')[0].replace('/', '')

        assert len(fen) == 64

        # Update board_array according to fen
        for board_rank in range(8):
            for board_file in range(8):
                square_value = fen[board_rank * 8 + board_file]
                if square_value.isalpha():
                    board_array[self.config.piece_index_map[square_value], board_rank, board_file] = 1

        assert board_array.shape == (12, 8, 8)
        return board_array

    def fen_to_castling_state(self, fen: str):
        castling_state = np.zeros(shape=(4, 8, 8), dtype=np.float32)

        castling_rights = fen.split(' ')[2]
        if castling_rights[0] != '-':
            for c in castling_rights:
                castling_state[self.config.castling_index_map[c], :, :] = 1

        return castling_state

    @staticmethod
    def fen_to_en_passant_state(fen: str):
        en_passant_state = np.zeros(shape=(1, 8, 8))
        en_passant_square = fen.split(' ')[3]
        if en_passant_square != '-':
            rank_index, file_index = ChessGameParser.square_to_coordinate(en_passant_square)
            en_passant_state[0, rank_index, file_index] = 1

        return en_passant_state

    @staticmethod
    def parse_time_control(time_control: str):
        """
        Parse the time control into approximate game length, given as time + increment*40, as 40 is the average
        number of moves in chess games.

        Then normalize on 0 to 1, with 1 being standard time controls calculated as 90 * 60 + 30 * 40

        :param time_control: a string in the form of 'time+increment', both given in seconds
        :return: An int in seconds of estimated total length
        """
        normalization_max = 90 * 60 + 30 * 40
        if time_control == '-':
            return normalization_max

        tc = time_control.split('+')
        clock, increment = tc[0], 0

        if len(tc) > 1:
            increment = tc[1]

        seconds = int(clock) + int(increment) * 40

        return min(seconds / normalization_max, 1)

    @staticmethod
    def parse_result(result: str):
        """
        Parse the game result from the result string
        :param result: A string in '1-0' for white win, '0-1' for black win, '1/2-1/2' for a draw
        :return: 1 for white win, 0 for draw, -1 for black win
        """
        result_map = {
            '1-0': 1,
            '1/2-1/2': 0,
            '0-1': -1,
        }

        return result_map[result]

    @staticmethod
    def black_to_move(fen: str):
        return fen.split(' ')[1] == 'b'

    @staticmethod
    def flip_fen(fen: str):
        """
        Flip the fen, so black becomes white and then the board is from white's perspective. When flipping the board,
        reflect the rows rather than rotating. This way, the king side will stay on the right and queen side on the
        left, as it would be for white. I.e., make the position equivalent to if white had arrive there.

        :param fen: The current state string in FEN format
        :return: The flipped state string in FEN format
        """
        # Deconstruct the fen
        board_state, to_move, castling_rights, en_passant_target, halfmove_clock, move_number = fen.split(' ')

        # Swap black for white pieces and reverse rows in the board_state
        board_state = ''.join([square.lower() if square.isupper() else square.upper() for square in board_state])
        board_state = '/'.join(reversed(board_state.split('/')))

        # Swap whose turn to move
        to_move = 'w' if to_move == 'b' else 'b'

        # Swap castling rights
        castling_rights = ''.join(sorted([c.lower() if c.isupper() else c.upper() for c in castling_rights],
                                         key=lambda c: {'K': 0, 'Q': 1, 'k': 2, 'q': 3, '-': 4}[c]))

        # Swap en passant square if necessary by flipping the rank
        if en_passant_target != '-':
            en_passant_target = en_passant_target[0] + str(int(-int(en_passant_target[1]) + 9))

        return ' '.join([board_state, to_move, castling_rights, en_passant_target, halfmove_clock, move_number])

    @staticmethod
    def square_to_coordinate(square: str):
        """
        Take a square and map to a tuple of (rank_idx, file_idx) within the canonical representation. Note that rank
        8 is at index 0, rank 7 is at index 1, ... , rank 1 is at index 7.

        E.g. 'a1' maps to (7, 0), 'e5' maps to (3, 4)

        :param square: String representation of the square from fen
        :return: tuple of (rank_index, file_index)
        """
        rank_index = 8 - int(square[1])
        file_index = ord(square[0]) - ord('a')
        return rank_index, file_index

    @staticmethod
    def normalize_elo(elo):
        """
        Normalize elo on the range of -1 to 1, assuming min elo is 0 and max elo is 3000
        """
        return elo/1500 - 1

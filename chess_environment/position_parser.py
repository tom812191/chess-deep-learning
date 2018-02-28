import numpy as np

import config


class ChessPositionParser:
    """
    Utility to convert games loaded from chess.pgn.read_game to training examples for the policy and value network.

    8x8 board with 18 channels
    The 18 channels are:
        12 for piece type: KQRBNPkqrbnp
         4 for castling KQkq
         1 for en passant
         1 player elo
    """

    def __init__(self, cfg: config.Config, fens: list, elos: list, fens_have_counters=False):
        """
        Create the chess game
        :param fens: List of fen strings
        :param elos: List of elos for player to move, same length as fens
        :param cfg: Config class
        :param fens_have_counters: Do the fen strings have the halfmove and fullmove counters
        """
        self.config = cfg
        self.fens = fens if fens_have_counters else ChessPositionParser.full_fens(fens)
        self.elos = elos

        assert len(fens) == len(elos)

    def get_canonical_input(self):
        """
        Create the canonical input for the fen values
        """

        states = []
        for fen, elo in zip(self.fens, self.elos):

            if ChessPositionParser.black_to_move(fen):
                fen = ChessPositionParser.flip_fen(fen)

            piece_state = self.fen_to_piece_state(fen)
            castling_state = self.fen_to_castling_state(fen)
            en_passant_state = self.fen_to_en_passant_state(fen)
            elo_state = np.full((1, 8, 8), elo)

            canonical_input = np.vstack((
                piece_state,
                castling_state,
                en_passant_state,
                elo_state,
            ))

            assert canonical_input.shape == (18, 8, 8)

            states.append(canonical_input)

        states = np.array(states)
        assert states.shape == (len(self.fens), 18, 8, 8)
        return states

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
            rank_index, file_index = ChessPositionParser.square_to_coordinate(en_passant_square)
            en_passant_state[0, rank_index, file_index] = 1

        return en_passant_state

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

    @staticmethod
    def full_fens(fens):
        """
        Take a list of fens without move counters and add in the move counters
        """
        return [
            ' '.join([fen, '0' if fen.split(' ')[1] == 'w' else '1', '1'])
            for fen in fens
        ]

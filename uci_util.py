from itertools import product


def all_uci_labels():
    """
    Create an exhaustive list of Universal Chess Interface (UCI) moves. These labels will be the output for the policy
    portion of the neural network.

    UCI moves are just given by the origin square and the destination square.
    Examples: e2e4, e7e5, e1g1 (white short castling), e7e8q (for promotion)
    """
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    promotions = ['q', 'r', 'b', 'n']

    # Squares in cartesian coordinates if the origin is in the bottom left of the board. So the ranks map directly to
    # the y coordinate and the files map to the x coordinate, both indexed from 0
    squares = list(product(range(8), range(8)))

    legal_moves = []
    for origin, destination in product(squares, squares):
        if origin == destination:
            continue

        # If legal move, then push to legal moves
        legal_move_types = [
            # Same rank
            origin[1] == destination[1],

            # Same file
            origin[0] == destination[0],

            # Same diagonal
            abs(destination[1] - origin[1]) == abs(destination[0] - origin[0]),

            # Knight move
            destination in knight_moves(origin),
        ]

        if any(legal_move_types):
            move = files[origin[0]] + ranks[origin[1]] + files[destination[0]] + ranks[destination[1]]
            legal_moves.append(move)

            if destination in pawn_promotions(origin):
                for promotion in promotions:
                    legal_moves.append(move + promotion)
    return legal_moves


def knight_moves(origin):
    moves = [
        (origin[0] - 1, origin[1] + 2),
        (origin[0] + 1, origin[1] + 2),

        (origin[0] + 2, origin[1] - 1),
        (origin[0] + 2, origin[1] + 1),

        (origin[0] - 1, origin[1] - 2),
        (origin[0] + 1, origin[1] - 2),

        (origin[0] - 2, origin[1] - 1),
        (origin[0] - 2, origin[1] + 1),
    ]

    return [move for move in moves if (0 <= move[0] <= 7) and (0 <= move[1] <= 7)]


def pawn_promotions(origin):
    if origin[1] == 1:
        return [(file, 0) for file in [origin[0] - 1, origin[0], origin[0] + 1] if 0 <= file <= 7]

    if origin[1] == 6:
        return [(file, 7) for file in [origin[0] - 1, origin[0], origin[0] + 1] if 0 <= file <= 7]

    return []

"""
Implements Monte Carlo Tree Search for our agent
"""
from collections import defaultdict
import chess
import chess.uci
import numpy as np
import json
import pickle
import os

import config
from chess_environment.position_parser import ChessPositionParser
from chess_environment.engine import Stockfish
from agent import predict


class ChessSearchNode:
    """
    A node in the Monte Carlo Search Tree. All values are always stored from white's POV. That is, a positive
    value indicates that white is winning.
    """
    def __init__(self):
        self.children = defaultdict(ChessSearchNode)
        self.parent = None

        self.prior_probability = None

        self.child_move_indexes = None
        self.child_policy = None
        self.child_values = None

        self.move_index = None
        self.visits = 0
        self.static_value = None
        self._rollup_value = None
        self._move_probability = None

        self.children_created = False
        self.our_move = True
        self.depth = 0

    def expand(self, policy, values, move_indexes):
        if self.children_created:
            raise RuntimeError('Children already created for node')

        self.child_policy = policy
        self.child_values = np.array(values)
        self.child_move_indexes = move_indexes

        for child_index, move_index in enumerate(move_indexes):
            self.children[move_index].parent = self
            self.children[move_index].move_index = move_index
            self.children[move_index].static_value = values[child_index]
            self.children[move_index].prior_probability = policy[child_index]
            self.children[move_index].our_move = not self.our_move
            self.children[move_index].depth = self.depth + 1

            self.children_created = True

    @property
    def rollup_value(self):
        """
        Recursively calculate the rollup value for the node
        """
        if self._rollup_value is not None:
            return self._rollup_value

        if not self.children_created:
            self._rollup_value = self.static_value
            return self.static_value

        child_rollup_values = np.array([self.children[child_idx].rollup_value
                                        for child_idx in self.child_move_indexes])
        self._rollup_value = np.dot(child_rollup_values, self.child_policy)

        return self._rollup_value

    @property
    def move_probability(self):
        """
        The probability of taking the specific branch
        """
        if self._move_probability is not None:
            return self._move_probability

        if self.parent is None:
            self._move_probability = self.prior_probability
        else:
            self._move_probability = self.parent.move_probability * self.prior_probability

        return self._move_probability


class ChessMonteCarloTreeSearch:
    def __init__(self, cfg: config.Config, move_predictor: predict.MovePredictor, position_parser: ChessPositionParser,
                 num_simulations=None, fen=chess.STARTING_FEN, deterministic=False, stockfish=None,
                 player_elo=1500, opponent_elo=1500, rating_transform_const=0.3, deterministic_moves=None):
        """
        Implement monte carlo tree search over possible chess moves.

        :param cfg: Global configuration
        :param position_parser: a ChessPositionParser instance that already contains elo and time control info
        :param deterministic: If true, select the move deterministically, otherwise select probabilistically
        """
        self.config = cfg
        self.move_predictor = move_predictor
        self.position_parser = position_parser
        self.num_simulations = num_simulations or self.config.play.num_simulations
        self.deterministic = deterministic
        self.deterministic_moves = deterministic_moves if deterministic_moves is not None else {}

        self.player_elo = player_elo
        self.opponent_elo = opponent_elo
        self.rating_transform_const = rating_transform_const

        self.move_map = {chess.Move.from_uci(move): idx for idx, move in enumerate(self.config.labels)}

        self.board = chess.Board(fen)

        self.stockfish = stockfish or Stockfish(self.config)

        self.root = ChessSearchNode()
        self.is_white = self.board.turn == chess.WHITE
        self.root_move_counter = int(fen.split(' ')[-1]) + (0 if self.is_white else 0.5)

        self.move_probabilities = None

        # Cache for predictions from the neural network and stockfish valuations
        self.cache_path = self.config.play.mcts_cache_path
        self.prediction_cache = pickle.load(open(self.cache_path, 'rb')) if os.path.exists(self.cache_path) else {}
        self.prediction_cache_size = self.config.play.mcts_cache_size

    @property
    def current_state(self):
        return self.board.fen()

    @property
    def our_move(self):
        return (self.board.turn == chess.WHITE) and self.is_white

    def set_position(self, fen, player_elo=None, opponent_elo=None, num_simulations=None,
                     rating_transform_const=None, deterministic_moves=None):
        self.board = chess.Board(fen)
        self.position_parser.reset(fens=[fen], elos=[self.player_elo], fens_have_counters=True)
        self.root = ChessSearchNode()
        self.move_probabilities = None
        self.is_white = self.board.turn == chess.WHITE
        self.root_move_counter = int(fen.split(' ')[-1]) + (0 if self.is_white else 0.5)

        if player_elo is not None:
            self.player_elo = player_elo

        if opponent_elo is not None:
            self.opponent_elo = opponent_elo

        if num_simulations is not None:
            self.num_simulations = num_simulations

        if rating_transform_const is not None:
            self.rating_transform_const = rating_transform_const

        if deterministic_moves is not None:
            self.deterministic_moves = deterministic_moves

        return self

    def run_search(self):
        """
        Run the monte carlo tree search to find the best next move
        """
        # Create the root node
        value = self.stockfish_eval_to_value(self.stockfish.eval(self.board))
        self.root.static_value = value
        self.root.prior_probability = 1.0

        # Expand the first set of leaf nodes
        policy, values, move_indexes = self.expand_node()
        self.root.expand(policy, values, move_indexes)

        # Run simulations
        for _ in range(self.num_simulations):
            self.search_node(self.root)

        return self

    def search_node(self, node: ChessSearchNode):
        """
        Search the current tree node.

        Return the value from the perspective of the player to play, where positive value is winning
        """

        # Check if game is over
        if self.board.result() != '*':
            return node.static_value

        # If at a leaf node, expand
        if not node.children_created:
            policy, values, move_indexes = self.expand_node()
            node.expand(policy, values, move_indexes)
            return node.static_value

        # Select the next node
        move_index = self.select_child_stochastic(node)

        # Search next move
        self.board.push(chess.Move.from_uci(self.config.labels[move_index]))
        value = self.search_node(node.children[move_index])
        self.board.pop()

        return value

    def expand_node(self):
        """
        Expand the current board position. Get the prior move probabilities and next position values for all
        legal moves.
        """
        fen = self.board.fen()
        partial_fen = ' '.join(fen.split(' ')[:-2])
        elo = self.player_elo if self.our_move else self.opponent_elo
        moves = self.move_predictor.predict(fen, elo, fen_has_counters=True, elo_is_normalized=False)

        if partial_fen in self.deterministic_moves:
            return self.expand_node_deterministic(partial_fen, moves)

        policy, values, legal_move_indexes = [], [], []
        for move, data in moves.items():
            policy.append(data['probability'])
            values.append(data['value'])
            legal_move_indexes.append(self.move_map[move])

        return policy, values, legal_move_indexes

    def expand_node_deterministic(self, partial_fen, moves):
        """
        Expand the node with a move from a predetermined move tree
        """
        san = self.deterministic_moves[partial_fen]
        move = self.board.parse_san(san)

        policy, values, legal_move_indexes = [1.0], [moves[move]['value']], [self.move_map[move]]
        return policy, values, legal_move_indexes

    @staticmethod
    def select_child_stochastic(node: ChessSearchNode):
        """
        Stochastically select a child with probabilities weighted by prior and value
        """
        policy = np.array(node.child_policy)
        pi = policy / policy.sum()
        return np.random.choice(node.child_move_indexes, p=pi)

    def get_json(self, max_depth=6):
        """
        Convert the results of the search tree to JSON.
        """
        def process_tree(node: ChessSearchNode, board: chess.Board):
            if node.move_index is None:
                fen = board.fen()
                move_san = ''
            else:
                move = chess.Move.from_uci(self.config.labels[node.move_index])
                move_san = board.san(move)
                board.push(move)
                fen = board.fen()

            tree = {
                'fen': fen,
                'prev_move': move_san,

                'static_value': node.static_value,
                'rollup_value': node.rollup_value,
                'move_probability': node.move_probability,
            }

            if node.children_created and node.depth < max_depth:
                children = [process_tree(child, board) for move_index, child in node.children.items()]
                children = sorted(children, key=lambda child: child['move_probability'], reverse=True)

                tree['children'] = children

            if node.depth > 0:
                board.pop()

            return tree

        board = self.board.copy()
        tree = process_tree(self.root, board)
        return json.dumps(tree, indent=2)

    def get_evals(self):
        return self.root.static_value, self.root.rollup_value

    @staticmethod
    def stockfish_eval_to_value(evaluation, k=0.6):
        """
        Convert the stockfish evaluation in the range of roughly -100 to +100 to -1 to +1 using
        a logistic function
        """
        return 2/(1 + np.exp(-k * evaluation)) - 1

    @staticmethod
    def value_to_stockfish_eval(value, k=0.6):
        """
        Inverse of stockfish value conversion
        """
        if value >= 1:
            return 100
        elif value <= -1:
            return -100

        return (-1 / k) * np.log(2/(1 + value) - 1)

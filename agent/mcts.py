"""
Implements Monte Carlo Tree Search for our agent
"""
from collections import defaultdict
import chess
import chess.uci
import numpy as np

import config
from chess_environment.game_parser import ChessPositionParser
from chess_environment.engine import Stockfish


class ChessMonteCarloTreeSearch:
    def __init__(self, cfg: config.Config, model, position_parser: ChessPositionParser, num_simulations=None,
                 fen=chess.STARTING_FEN, deterministic=False, tau=None):
        """
        Implement monte carlo tree search over possible chess moves.

        :param cfg: Global configuration
        :param model: Policy/value deep neural network
        :param position_parser: a ChessPositionParser instance that already contains elo and time control info
        :param deterministic: If true, select the move deterministically, otherwise select probabilistically
        """
        self.config = cfg
        self.model = model
        self.position_parser = position_parser
        self.num_simulations = num_simulations or self.config.play.num_simulations
        self.deterministic = deterministic
        self.tau = tau

        # Upper confidence tree score constant
        self.ucts_const = self.config.play.upper_confidence_tree_score_constant

        self.move_map = {chess.Move.from_uci(move): idx for idx, move in enumerate(self.config.labels)}

        self.board = chess.Board(fen)

        self.stockfish = Stockfish(self.config)

        self.tree = defaultdict(ChessSearchNode)

        self.is_white = self.board.turn == chess.WHITE

        self.move_probabilities = None

    @property
    def current_state(self):
        return self.board.fen()

    def set_position(self, fen):
        self.board = chess.Board(fen)
        self.position_parser.reset(fen=fen)
        self.tree = defaultdict(ChessSearchNode)
        self.move_probabilities = None
        return self

    def get_next_move(self):
        """
        Select the next move
        """
        if self.move_probabilities is None:
            self.search_moves()
            self.calc_move_probabilities()

        if self.deterministic:
            next_move_idx = np.argmax(self.move_probabilities)
            return self.config.labels[next_move_idx]

        return np.random.choice(self.config.labels, p=self.move_probabilities)

    def search_moves(self):
        """
        Run the monte carlo tree search to find the best next move
        """
        for _ in range(self.num_simulations):
            print(_)
            self.search_node(is_opponent=False)

        return self

    def search_node(self, is_opponent=False):
        """
        Search the current tree node
        """
        # Check if game is over
        result = self.board.result()
        if result != '*':
            result_map = {'1-0': 1, '1/2-1/2': 0, '0-1': -1}
            return result_map[result] * (-1 if is_opponent else 1) * (1 if self.is_white else -1)

        state = self.current_state

        if state not in self.tree:
            policy, value, move_indexes = self.expand_node()
            self.tree[state].set_predictions(policy, value, move_indexes)
            return value

        # Select the next node
        move_index = self.select_node()

        # Update statistics
        current_node = self.tree[state]
        current_node.total_visits += 1

        # Search next move
        self.board.push(chess.Move.from_uci(self.config.labels[move_index]))
        next_value = -self.search_node(is_opponent=(not is_opponent))
        node_stats = current_node.children[move_index]
        node_stats.num_visits += 1
        node_stats.accumulated_value += next_value
        self.board.pop()

        return next_value

    def expand_node(self):
        """
        Expand the current board position
        """
        legal_move_indexes = [self.move_map[m] for m in self.board.legal_moves]
        policy, value = self.predict(legal_move_indexes)
        return policy, value, legal_move_indexes

    def select_node(self):
        """
        Select the next node to explore based on the upper confidence bound
        """
        state = self.current_state
        current_node = self.tree[state]

        if not current_node.children_created:
            current_node.create_children()

        best_move_idx = None
        best_score = -1000
        for move_idx, move_stats in current_node.children.items():
            uct_score = self.upper_confidence_tree_score(move_stats, current_node.total_visits)
            if uct_score > best_score:
                best_score = uct_score
                best_move_idx = move_idx

        return best_move_idx

    def predict(self, legal_moves_indexes):
        """
        Get the policy and value from the neural network for the current board state
        """
        # Get outputs from neural network
        policy = self.model.predict(self.position_parser.input_tensor)[0]
        value = self.stockfish.eval(self.board) / 10

        if not self.is_white:
            value *= -1

        # Mask illegal moves
        masked_policy = policy[legal_moves_indexes]
        policy = masked_policy / masked_policy.sum()

        return policy, value

    def upper_confidence_tree_score(self, node_stats, parent_visits):
        """
        Get the upper confidence tree score for the current node
        """
        return node_stats.accumulated_value / (1 + node_stats.num_visits) + \
               self.ucts_const * node_stats.prior_probability * np.sqrt((1 + parent_visits) / (2 + node_stats.num_visits))

    def calc_move_probabilities(self):
        """
        Get the pi vector of move probabilities based on node visit frequency
        """
        state = self.current_state
        node = self.tree[state]

        self.move_probabilities = np.zeros(self.config.n_labels)

        for move_idx, child in node.children.items():
            self.move_probabilities[move_idx] = child.num_visits

        self.move_probabilities /= np.sum(self.move_probabilities)
        if self.tau is not None:
            self.move_probabilities = np.power(self.move_probabilities, 1 / self.tau)
            self.move_probabilities /= np.sum(self.move_probabilities)


class ChessSearchNode:
    """
    A node in the Monte Carlo Search Tree
    """
    def __init__(self):
        self.children = defaultdict(ChessNodeStats)
        self.total_visits = 0

        self.children_policy = None
        self.children_move_indexes = None
        self.network_value = 0

        self.children_created = False

    def set_predictions(self, children_policy, network_value, children_move_indexes):
        self.children_policy = children_policy
        self.network_value = network_value
        self.children_move_indexes = children_move_indexes

    def create_children(self):
        for idx, move_idx in enumerate(self.children_move_indexes):
            self.children[move_idx].prior_probability = self.children_policy[idx]
        self.children_created = True


class ChessNodeStats:
    """
    A node in the Monte Carlo Search Tree
    """
    def __init__(self):
        self.num_visits = 0
        self.accumulated_value = 0
        self.prior_probability = 0

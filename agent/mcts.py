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


class ChessSearchNode:
    """
    A node in the Monte Carlo Search Tree
    """
    def __init__(self):
        self.children = defaultdict(ChessSearchNode)
        self.parent = None

        self.prior_probability = None

        self.child_move_indexes = None
        self.child_policy = None
        self.child_values = None  # Values from this node's POV

        self.move_index = None
        self.visits = 0
        self.static_value = None
        self.accumulated_value = 0
        self.average_value = 0

        self.children_created = False
        self.our_move = True
        self.depth = 0

    def expand(self, policy, values, move_indexes):
        if self.children_created:
            raise RuntimeError('Children already created for node')

        self.child_policy = policy
        self.child_values = -1 * np.array(values)
        self.child_move_indexes = move_indexes

        for child_index, move_index in enumerate(move_indexes):
            self.children[move_index].parent = self
            self.children[move_index].move_index = move_index
            self.children[move_index].static_value = values[child_index]
            self.children[move_index].prior_probability = policy[child_index]
            self.children[move_index].our_move = not self.our_move
            self.children[move_index].depth = self.depth + 1

            self.children_created = True

    def accumulate_value(self, value):
        self.accumulated_value += value
        self.average_value = self.accumulated_value / self.visits


class ChessMonteCarloTreeSearch:
    def __init__(self, cfg: config.Config, model, position_parser: ChessPositionParser, num_simulations=None,
                 ucts_const=None, fen=chess.STARTING_FEN, deterministic=False, tau=None, stockfish=None,
                 player_elo=1500, opponent_elo=1500, rating_transform_const=0.3):
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

        self.player_elo = player_elo
        self.opponent_elo = opponent_elo
        self.rating_transform_const = rating_transform_const

        # Upper confidence tree score constant
        self.ucts_const = ucts_const or self.config.play.upper_confidence_tree_score_constant

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

    def set_position(self, fen, num_simulations=None, tau=None, ucts_const=None):
        self.board = chess.Board(fen)
        self.position_parser.reset(fens=[fen], fens_have_counters=True)
        self.root = ChessSearchNode()
        self.move_probabilities = None
        self.is_white = self.board.turn == chess.WHITE
        self.root_move_counter = int(fen.split(' ')[-1]) + (0 if self.is_white else 0.5)

        if num_simulations is not None:
            self.num_simulations = num_simulations

        if tau is not None:
            self.tau = tau

        if ucts_const is not None:
            self.ucts_const = ucts_const

        return self

    def get_mcts_policy(self):
        """
        Calculate the new policy based on the MCTS
        """
        if self.move_probabilities is None:
            self.run_search()
            self.calc_move_probabilities()

        pickle.dump(self.prediction_cache, open(self.cache_path, 'wb'))

        return self.move_probabilities

    def get_next_move(self):
        """
        Select the next move
        """
        mcts_policy = self.get_mcts_policy()

        if self.deterministic:
            next_move_idx = np.argmax(mcts_policy)
            return self.config.labels[next_move_idx]

        return np.random.choice(self.config.labels, p=mcts_policy)

    def run_search(self):
        """
        Run the monte carlo tree search to find the best next move
        """
        # Create the root node
        value = self.stockfish_eval_to_value(self.stockfish.eval(self.board))
        self.root.static_value = value
        self.root.prior_probability = 1.0

        # Expand the first set of leaf nodes
        policy, values, move_indexes = self.expand_node(self.root)
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
            node.visits += 1
            node.accumulate_value(node.static_value)
            return node.average_value

        # If at a leaf node, expand
        if not node.children_created:
            policy, values, move_indexes = self.expand_node(node)
            node.expand(policy, values, move_indexes)
            return node.static_value

        node.visits += 1

        # Select the next node
        move_index = self.select_child_stochastic(node)

        # Search next move
        self.board.push(chess.Move.from_uci(self.config.labels[move_index]))
        value = -self.search_node(node.children[move_index])
        node.accumulate_value(value)
        self.board.pop()

        return value

    def expand_node(self, node):
        """
        Expand the current board position. Get the prior move probabilities and next position values for all
        legal moves.
        """
        legal_move_indexes = [self.move_map[m] for m in self.board.legal_moves]
        policy, values = self.predict(self.board.legal_moves, legal_move_indexes, node.depth + self.root_move_counter)
        return policy, values, legal_move_indexes

    def select_child(self, node):
        """
        Select the next node to explore based on the upper confidence bound
        """

        best_move_idx = None
        best_score = -1000
        for move_idx, child in node.children.items():
            uct_score = self.upper_confidence_tree_score(child)
            if uct_score > best_score:
                best_score = uct_score
                best_move_idx = move_idx

        return best_move_idx

    def select_child_stochastic(self, node: ChessSearchNode):
        """
        Stochastically select a child with probabilities weighted by prior and value
        """
        r_exp = self.rating_exponent_transform(self.player_elo if node.our_move else self.opponent_elo)
        v_prime = (np.max(node.child_values) - node.child_values) / 2
        p_prime = node.child_policy * np.power(1 - v_prime, r_exp)
        pi = p_prime / p_prime.sum()

        return np.random.choice(node.child_move_indexes, p=pi)

    def rating_exponent_transform(self, rating):
        return self.rating_transform_const * np.sqrt(rating)

    def predict(self, legal_moves, legal_moves_indexes, move_counter):
        """
        Get the policy and value from the neural network for the current board state.

        Value will be returned from the point of view of the player to move. E.g. if black is winning and it is black
        to move, then value will be positive
        """

        fen = self.board.fen()[:-2]

        # Get the policy for legal follow up moves
        if fen in self.prediction_cache:
            policy = self.prediction_cache[fen]['policy']
            values = self.prediction_cache[fen]['values']
            return policy, values

        policy = self.model.predict(self.position_parser.get_canonical_input())[0]

        # Mask illegal moves
        masked_policy = policy[legal_moves_indexes]
        policy = masked_policy / masked_policy.sum()

        values = []
        for move in legal_moves:
            self.board.push(move)
            values.append(self.stockfish_eval_to_value(self.stockfish.eval(self.board)))
            self.board.pop()

            # Cache the value
            self.prediction_cache[fen] = {
                'policy': policy,
                'values': values,
                'move_counter': move_counter,
            }

            if len(self.prediction_cache) > self.prediction_cache_size:
                self.reduce_cache()

        return policy, values

    def upper_confidence_tree_score(self, node: ChessSearchNode):
        """
        Get the upper confidence tree score for the current node
        """
        value = -1 * (node.average_value if node.visits > 0 else node.static_value)
        return value + self.ucts_const * node.prior_probability * \
                               np.sqrt(np.log(node.parent.visits) / (1 + node.visits))

    def calc_move_probabilities(self):
        """
        Get the pi vector of move probabilities based on node visit frequency
        """
        self.move_probabilities = np.zeros(self.config.n_labels)

        for move_idx, child in self.root.children.items():
            self.move_probabilities[move_idx] = child.visits

        self.move_probabilities /= np.sum(self.move_probabilities)
        if self.tau is not None:
            self.move_probabilities = np.power(self.move_probabilities, 1 / self.tau)
            self.move_probabilities /= np.sum(self.move_probabilities)

    def reduce_cache(self):
        """
        Delete cache for old positions and lines far ahead
        """
        self.prediction_cache = {key: value for key, value in self.prediction_cache.items()
                                 if (value['move_counter'] <= self.root_move_counter)
                                 or (value['move_counter'] > self.root_move_counter + 5)}

    def get_json(self, max_depth=6, agg_empty_leaves=False):
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

            value_factor = 1 if node.our_move else -1

            tree = {
                'fen': fen,
                'prev_move': move_san,

                'static_value': float(node.static_value) * value_factor,
                'stockfish_value': float(self.value_to_stockfish_eval(node.static_value)) * value_factor,
                'visits': int(node.visits),

                'average_value': float(node.average_value) * value_factor,
                'prior_probability': float(node.prior_probability),
                'posterior_probability': float(node.visits / node.parent.visits) if node.parent and node.parent.visits > 0 is not None else 1,
            }

            if node.children_created and node.depth < max_depth:
                children = [process_tree(child, board) for move_index, child in node.children.items()]
                children = sorted(children, key=lambda child: child['visits'], reverse=True)

                """
                if agg_empty_leaves:
                    leaves = [c for c in children if c['total_visits'] == 1]
                    if len(leaves) > 0:
                        agg_leaf = {
                            'prev_move': ', '.join([l['prev_move'] for l in leaves][:5]) + ('...' if len(leaves) > 5 else ''),
                            'static_value': np.mean(np.array([l['static_value'] for l in leaves])),
                            'total_visits': len(leaves),
                            'rollup_value': 0,
                            'rollup_visits': 0,
                            'avg_value': 0,
                            'prior_probability': sum([l['prior_probability'] for l in leaves]),
                            'posterior_probability': len(leaves) / parent_visits,
                        }
                        children = [c for c in children if c['total_visits'] > 1] + [agg_leaf]
                        
                """

                tree['children'] = children

            if node.depth > 0:
                board.pop()

            return tree

        board = self.board.copy()
        tree = process_tree(self.root, board)
        return json.dumps(tree)

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

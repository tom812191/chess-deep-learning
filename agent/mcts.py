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
from typing import Optional

import config
from chess_environment.position_parser import ChessPositionParser
from chess_environment.engine import Stockfish


class ChessMonteCarloTreeSearch:
    def __init__(self, cfg: config.Config, model, position_parser: ChessPositionParser, num_simulations=None,
                 ucts_const=None, fen=chess.STARTING_FEN, deterministic=False, tau=None, stockfish=None):
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
        self.ucts_const = ucts_const or self.config.play.upper_confidence_tree_score_constant

        self.move_map = {chess.Move.from_uci(move): idx for idx, move in enumerate(self.config.labels)}

        self.board = chess.Board(fen)

        self.stockfish = stockfish or Stockfish(self.config)

        self.tree = defaultdict(ChessSearchNode)

        self.is_white = self.board.turn == chess.WHITE

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
        self.tree = defaultdict(ChessSearchNode)
        self.move_probabilities = None
        self.is_white = self.board.turn == chess.WHITE

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
            self.search_moves()
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

    def search_moves(self):
        """
        Run the monte carlo tree search to find the best next move
        """
        for _ in range(self.num_simulations):
            self.search_node(is_opponent=False)

        return self

    def search_node(self, is_opponent=False):
        """
        Search the current tree node.

        Return the value from the perspective of the player to play, where positive value is winning
        """
        # Check if game is over
        result = self.board.result()
        if result == '1/2-1/2':
            return 0
        elif result != '*':
            # Player to move has lost. If game is over and not a draw, then previous move was winning move, so
            # this player lost
            return -10

        state = self.current_state

        if state not in self.tree:
            policy, value, move_indexes, children_static_values = self.expand_node()
            self.tree[state].set_predictions(policy, value, move_indexes, children_static_values)
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
        children_static_values = self.predict_values(self.board.legal_moves)
        return policy, value, legal_move_indexes, children_static_values

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
        Get the policy and value from the neural network for the current board state.

        Value will be returned from the point of view of the player to move. E.g. if black is winning and it is black
        to move, then value will be positive
        """

        fen = self.board.fen()[:-2]
        if fen in self.prediction_cache:
            self.prediction_cache[fen]['visits'] += 1
            return self.prediction_cache[fen]['policy'], self.prediction_cache[fen]['value']

        # Get outputs from neural network
        policy = self.model.predict(self.position_parser.get_canonical_input())[0]
        value = self.stockfish.eval(self.board) / 10

        # Mask illegal moves
        masked_policy = policy[legal_moves_indexes]
        policy = masked_policy / masked_policy.sum()

        # Cache the value
        self.prediction_cache[fen] = {
            'policy': policy,
            'value': value,
            'visits': 1,
        }

        if len(self.prediction_cache) > self.prediction_cache_size:
            self.reduce_cache()

        return policy, value

    def predict_values(self, moves):
        """
        Get static values from stockfish for leaf nodes
        """
        values = []
        for move in moves:
            self.board.push(move)
            values.append(-1 * self.stockfish.eval(self.board) / 10)
            self.board.pop()

        return values

    def upper_confidence_tree_score(self, node_stats, parent_visits):
        """
        Get the upper confidence tree score for the current node
        """
        accumulated_value = node_stats.accumulated_value if node_stats.num_visits > 0 else node_stats.static_value
        return accumulated_value / (1 + node_stats.num_visits) + \
               self.ucts_const * node_stats.prior_probability * np.sqrt(np.log(parent_visits) / (1 + node_stats.num_visits))

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

    def reduce_cache(self):
        """
        Delete half of the cache with the fewest visits
        """
        print('Cache purge')
        median = np.median(np.array([value['visits'] for _, value in self.prediction_cache.items()]))
        self.prediction_cache = {key: value for key, value in self.prediction_cache.items() if value['visits'] > median}

    def get_json(self, max_depth=6, agg_empty_leaves=False):
        """
        Convert the results of the search tree to JSON. The JSON will look like:

        {
            fen: "xxx",
            static_value: xxx,
            rollup_value: xxx,
            visits: xxx,
            prior_probability,
            children: [{
                fen: xxx,
                move: xxx,

                static_value: xxx,
                total_visits: xxx,

                rollup_value: xxx,
                rollup_visits: xxx,
                prior_probability: xxx,

                children: [...],
            }, ...],
        }
        """
        def process_tree(board: chess.Board, move_uci: Optional[str],
                         prior_probability: float, current_depth: int, parent_visits=None):
            if move_uci is None:
                fen = board.fen()
                move_san = ''
            else:
                move = chess.Move.from_uci(move_uci)
                move_san = board.san(move)
                board.push(move)
                fen = board.fen()

            node = self.tree[fen]

            if parent_visits is None:
                parent_visits = node.total_visits

            rollup_value = sum([child.accumulated_value for _, child in node.children.items()])
            rollup_value *= (-1 if current_depth % 2 else 1) * 10
            rollup_visits = sum([child.num_visits for _, child in node.children.items()])

            tree = {
                'fen': fen,
                'prev_move': move_san,

                'static_value': node.static_value * 10 * (-1 if current_depth % 2 else 1),
                'total_visits': node.total_visits,

                'rollup_value': rollup_value,
                'rollup_visits': rollup_visits,
                'avg_value': rollup_value / rollup_visits if rollup_visits > 0 else 0,
                'prior_probability': prior_probability,
                'posterior_probability': node.total_visits / parent_visits,
            }

            if node.children_created and current_depth < max_depth:
                children = [
                    process_tree(board, self.config.labels[move_index],
                                 float(node.children[move_index].prior_probability), current_depth + 1, parent_visits=node.total_visits)
                    for move_index in node.children_move_indexes if node.children[move_index].num_visits > 0]
                children = sorted(children, key=lambda child: child['total_visits'], reverse=True)

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

                tree['children'] = children

            if current_depth > 0:
                board.pop()

            return tree

        board = self.board.copy()
        tree = process_tree(board, None, 1.0, 0)
        return json.dumps(tree)


class ChessSearchNode:
    """
    A node in the Monte Carlo Search Tree
    """
    def __init__(self):
        self.children = defaultdict(ChessNodeStats)
        self.total_visits = 1

        self.children_policy = None
        self.children_move_indexes = None
        self.children_static_values = None
        self.static_value = 0

        self.children_created = False

    def set_predictions(self, children_policy, static_value, children_move_indexes, children_static_values):
        self.children_policy = children_policy
        self.static_value = static_value
        self.children_move_indexes = children_move_indexes
        self.children_static_values = children_static_values

    def create_children(self):
        for idx, move_idx in enumerate(self.children_move_indexes):
            self.children[move_idx].prior_probability = self.children_policy[idx]
            self.children[move_idx].static_value = self.children_static_values[idx]
        self.children_created = True


class ChessNodeStats:
    """
    A node's MCTS statistics
    """
    def __init__(self):
        self.num_visits = 0
        self.accumulated_value = 0
        self.prior_probability = 0
        self.static_value = 0

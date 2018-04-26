# chess-deep-learning
This project builds a new chess engine, inspired by AlphaZero. The goal, however, is to be useful to humans, not to play as strongly as possible.

## The Model
1. Prior Network: The first network takes the game state and the player elo rating and estimates the probability of the player making each legal move.
2. Probability Network: The second network takes the prior probabilities for each move from the first network along with Stockfish evaluations for depths 1, 3, 7, and 10 for each move. It outputs the probability that the player will make each move.
3. Evaluation: The game state is rolled out using stochastic Monte Carlo tree search, with node selection weighted by probabilities from the prevous step. Stockfish evaluations of depth 10 on the leaf nodes are rolled up to get an expected evaluation for each non-leaf node.

### Network Architectures
Both networks use Keras with a Theano backend.

#### Prior Network
The prior network is a deep residual convolutional neural network.

The input is a tensor of shape (18, 8, 8). The 8x8 is to represent the chess board. The 18 boards are
* 12: one-hot encode the presence of each piece type (pawn, rook, knight, bishop, king, queen, for white and black)
* 4: castling rights (king side and queen side for black and white)
* 1: En Passant square(s)
* 1: Player elo

The first layer is a convolutional layer with 256 filters of kernel size 5, followed by a batch normalization layer and a relu activation layer.

It has 7 residual blocks where each block consists of a convolutional layer, a batch normalization layer, a relu activation layer, another convolutional layer, another batch normalization layer, and another relu activation layer. Each convolutional layer has 256 filters of kernel size 3.

The output layer is a dense layer with softmax activation with a node for each move in the move space.

L2 regularization is used with a coefficient of 1e-4.

#### Probability Network
The probability network is a basic artificial neural network.

The input has 101 fields:
* 20 for candidate move prior probabilities (from the prior network)
* 80 for Stockfish evaluations (depth 1, 3, 7, 10 for each of the 20 candidate moves)
* 1 for player elo

The network then has two dense layers with 128 nodes each with sigmoid activations.

The output layer has 20 nodes (1 for each candidate move) with softmax activation.

L2 regularization is used with a coefficient of 7e-5.

### Training Data
All training data is taken from the [Lichess Games Database](https://database.lichess.org/).

Unique positions from games were aggregated and the move taken from that position stored for 4 different skill levels. These next moves form the target probability distribution.

Training was done on an AWS EC2 p3.2xlarge instance, which has 1 Tesla V100 GPU.

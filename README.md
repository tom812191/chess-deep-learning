# chess-deep-learning

## Road Map
#### 1. Organize Lichess database by position
* Create a table that gives unique positions, with a column that gives the next move that was made and the strength of the player who made it. Use this probability distribution to train policy vectors. 
#### 2. Retrain network
* Remove opponent elo from training. 
* Remove value component, just use stockfish evaluations.
#### 3. Test network
* Test network policy against actual move frequency
* Test MCTS move probabilities against actual move frequency
** Try fitting MCTS params based on player strength to better predict move probabilities

# chess-deep-learning

## Road Map
#### 1. Organize Lichess data into a database
#### 2. Change training strategy, with the goal being to teach a good policy, then dumb down for lower elo players
* Train on only strong grandmaster games and games with a one strong player against a weaker player. 
  * Assess model
* "Dumb down" the algorithm with lower elo games, to learn to differentiate weak play and strong play by elo
  * Assess model
#### 3. Implement MCTS on analysis
* Test on some actual games. Look at win probability when a player played our engines recommended move vs stockfish recommended move.

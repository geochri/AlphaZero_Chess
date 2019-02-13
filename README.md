# AlphaZero_Chess
# From-scratch implementation of AlphaZero for Chess

This repo demonstrates an implementation of AlphaZero framework for Chess, using python and PyTorch.

We all know that AlphaGo, created by DeepMind, created a big stir when it defeated reigning world champion Lee Sedol 4-1 in the game of Go in 2016, hence becoming the first computer program to achieve superhuman performance in an ultra-complicated game. 

However, AlphaGoZero, published (https://www.nature.com/articles/nature24270) a year later in 2017, push boundaries one big step further by achieving a similar feat without any human data inputs. A subsequent paper (https://arxiv.org/abs/1712.01815) released by the same group DeepMind successfully applied the same reinforcement learning + supervised learning framework to chess, outperforming the previous best chess program Stockfish after just 4 hours of training.

Inspired by the power of such supervised reinforcement learning models, I created a repository to build my own chess AI program from scratch, closely following the methods as described in the papers above.

# Contents
In this repository, you will find the following core scripts:

1) MCTS_chess.py - implements the Monte-Carlo Tree Search (MCTS) algorithm based on Polynomial Upper Confidence Trees (PUCT) method for leaf transversal. This generates datasets (state, policy, value) for neural network training

2) alpha_net.py - PyTorch implementation of the AlphaGoZero neural network architecture, with slightly reduced number of residual blocks (19) and convolution channels (256) for faster computation. The network consists of, in order:
- A convolution block with batch normalization
- 19 residual blocks with each block consisting of two convolutional layers with batch normalization
- An output block with two heads: a policy output head that consists of convolutional layer with batch normalization followed by logsoftmax, and a value head that consists of a convolutional layer with relu and tanh activation.

3) chess_board.py – Implementation of a chess board python class with all game rules and possible moves

4) encoder_decoder.py – list of functions to encode/decode chess board class for input/interpretation into neural network, as well as encode/decode the action policy output from neural network

5) evaluator.py – arena class to pit current neural net against the neural net from previous iteration, and keeps the neural net that wins the most games

6) train.py – function to start the neural network training process

7) train_multiprocessing.py – multiprocessing version of train.py

8) pipeline.py – script to starts a sequential iteration pipeline consisting of MCTS search to generate data and neural network training. The evaluator arena function is temporarily excluded here during the early stages of training the neural network.

9) visualize_board.py – miscellaneous function to visualize the chessboard in a more attractive way

10) analyze_games.py – miscellaneous script to visualize and save the chess games

# Iteration pipeline

A full iteration pipeline consists of:
1) Self-play using MCTS (MCTS_chess.py) to generate game datasets (game state, policy, value), with the neural net guiding the search by providing the prior probabilities in the PUCT algorithm

2) Train the neural network (train.py) using the (game state, policy, value) datasets generated from MCTS self-play

3) Evaluate (evaluator.py) the trained neural net (at predefined checkpoints) by pitting it against the neural net from the previous iteration, again using MCTS guided by the respective neural nets, and keep only the neural net that performs better.

4) Rinse and repeat. Note that in the paper, all these processes are running simultaneously in parallel, subject to available computing resources one has.

# How to play
1) Run pipeline.py to start the MCTS search and neural net training process. Change the folder and net saved names accordingly. Note that for the first time, you will need to create and save a random, initialized alpha_net for loading. Multiprocessing is enabled, which shares the PyTorch net model in a single CUDA GPU across 6 CPUs workers each running a MCTS self-play.

OR

1) Run the MCTS_chess.py to generate self-play datasets. Note that for the first time, you will need to create and save a random, initialized alpha_net for loading. Multiprocessing is enabled, which shares the PyTorch net model in a single CUDA GPU across 6 CPUs workers each running a MCTS self-play. 

2) Run train.py to train the alpha_net with the datasets.

3) At predetermined checkpoints, run evaluator.py to evaluate the trained net against the neural net from previous iteration. Saves the neural net that performs better. Multiprocessing is enabled, which shares the PyTorch net model in a single CUDA GPU across 6 CPUs workers each running a MCTS self-play. 

4) Repeat for next iteration.

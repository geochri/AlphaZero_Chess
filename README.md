# AlphaGoZero_Chess
# From-scratch implementation of AlphaGoZero for Chess

1) MCTS_chess.py - implements the Monte-Carlo Tree Search (MCTS) algorithm based on Polynomial Upper Confience Trees (PUCT) method for leaf transversal. This generates datasets (state, policy, value) for neural network training

2) alpha_net.py - PyTorch implementation of the AlphaGoZero neural network architecture, with slightly reduced number of residual blocks and convolution channels for faster computation. The network consists of, in order:
- A convolution block with batch normalization
- 13 residual blocks with each block consisting of two convolutional layers with batch normalization
- output block with two heads: a policy output head that consists of convolutional layer with batch normalization followed by logsoftmax, and a value head that consists of a convolutional layer with relu and tanh activation.

3) 

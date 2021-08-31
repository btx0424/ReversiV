# CS303A Fall-2020 Project: ReversiV
This project is a realization of [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) on the board game [Reversi](https://en.wikipedia.org/wiki/Reversi) with a drastically simplified model.

The competition only allowed one source file for submission, so I had to hard-code the parameters and implement the value network with Numpy.

The training took around 10 hours (with multi-process data collection) on an RTX2080. The final rank was 5/140+. 

[Report]()





![network.png](https://github.com/sustc11810424/ReversiV/blob/main/figures/network.png?raw=true)


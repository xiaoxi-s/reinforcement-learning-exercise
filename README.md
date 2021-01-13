# Reinforcement Learning Exercise

Algorithm and Exercise for reinforcement learning. The reference book is [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html). Below is a list of environments used for testing algorithms.

## Dynamic Programming

Environment: grid world (example 4.1 of the RL book).

## Monte Carlo

Environment: simplified Blackjack (example 5.1) 

## Temporal Difference

Environment: Windy gridworld (example 6.5)

## Function Approximation

Environment: Gym Mountain Car environment

## Policy Gradient

Environment: Cliff walking (example 6.6)

Remark: Reinforce with MC has very high variance and usually does not converge to the optimal policy (end up with going to the cliff all the time). While, the actor-critic algorithm would sometimes end up with the suboptimal policy. 
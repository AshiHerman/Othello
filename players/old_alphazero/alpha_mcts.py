import math
import random

import numpy as np
from alpha_net import PolicyP, PolicyV

# ------------ Configuration ------------

EPISODES_PER_MOVE = 1#100              # Number of MCTS rollouts per move
EXPLORATION_CONSTANT = math.sqrt(2)   # Exploration factor for UCT


# ------------ MCTS Node Definition ------------

class MctsNode:
    def __init__(self, problem, state, parent=None, action=None, prior=None):
        self.problem = problem
        self.state = state
        self.parent = parent
        self.action = action
        self.children = dict()  # action -> child node
        self.prior = prior      # action -> prior probability (from NN)
        self.N = dict()         # action -> visit count
        self.W = dict()         # action -> total value
        self.Q = dict()         # action -> mean value
        self.is_expanded = False
        self.value = 0
        self.visits = 0                            # Number of times this node was visited

    def __repr__(self):
        """
        Returns a debug-friendly string representation of the node.
        """
        return (f"MctsNode(action={self.action}, visits={self.visits}, "
                f"value={self.value:.2f}, untried={len(self.untried)})")

def select(self, c_puct):
    node = self
    while node.is_expanded:
        # Choose action maximizing PUCT
        best_a, _ = max(
            self.prior.keys(),
            key=lambda a: self.Q[a] + 
                c_puct * self.prior[a] * math.sqrt(sum(self.N.values())) / (1 + self.N[a])
        )
        if best_a not in node.children:
            # Not expanded
            return node, best_a
        node = node.children[best_a]
    return node, None


def expand(self, nn):
    # Call neural net to get priors & value for this node
    policy_logits, value = nn(self.state)
    priors = softmax_over_legal_moves(policy_logits, self.problem.actions(self.state))
    self.prior = {a: priors[a] for a in self.problem.actions(self.state)}
    for a in self.prior:
        self.N[a] = 0
        self.W[a] = 0
        self.Q[a] = 0
    self.is_expanded = True
    return value

def simulate(self, policy):
    """
    Simulates a random playout from this node's state to a terminal state and returns the outcome.
    """
    prob = self.problem
    state = self.state
    player = prob.player(state)  # Player whose turn is next

    while not prob.isEnd(state):
        actions = prob.actions(state)
        probs = np.array(policy[a] for a in actions)
        probs = probs / probs.sum()
        a = np.random(actions, p=probs)
        state = prob.enact(state, a)            # Transition to next state

    return prob.reward(state) * (1 if player == "O" else -1)  # Score from "O"'s perspective

def mcts_policy(self):
    child_vals = [child.value for child in self.children]
    return child_vals / child_vals.sum() # Wrong syntax

def backpropagate(self, reward):
    """
    Propagates the result of a simulation back up the tree, updating value and visit counts.
    """
    node = self
    stats = []
    while node is not None:
        node.value += reward
        node.visits += 1
        reward = -reward  # Alternate reward sign due to alternating turns
        node = node.parent
        stats.append((node.state, node.mcts_policy(), node.value))
    return stats


# ------------ MCTS Core Algorithm ------------

def mcts(problem, state, episodes=EPISODES_PER_MOVE, c=EXPLORATION_CONSTANT):
    """
    Runs Monte Carlo Tree Search starting from the given state for a fixed number of episodes.
    Returns the root of the resulting search tree.
    """
    policyP = PolicyP()
    policyV = PolicyV()
    root = MctsNode(problem, state)
    for _ in range(episodes):
        node = root.select(policyP, c)
        if node.untried:
            node = node.expand()
        result = node.simulate()
        node.backpropagate(result)
    return root


# ------------ Move Selection Interface ------------
class MCTS():
    def choose_move(self, problem, state, episodes=EPISODES_PER_MOVE, c=EXPLORATION_CONSTANT):
        """
        Runs MCTS from the current state and returns the action with the highest visit count.
        """
        root = mcts(problem, state, episodes, c)
        best = max(root.children, key=lambda ch: ch.visits)
        return best.action
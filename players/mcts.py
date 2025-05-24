import math
import random


# ------------ Configuration ------------

EPISODES_PER_MOVE = 100              # Number of MCTS rollouts per move
EXPLORATION_CONSTANT = math.sqrt(2)   # Exploration factor for UCT


# ------------ MCTS Node Definition ------------

class MctsNode:
    def __init__(self, problem, state, parent=None, action=None):
        """
        Initializes a Monte Carlo Tree Search (MCTS) node with the given state, parent, and action.
        """
        self.problem = problem                      # Reference to the problem (game)
        self.state = state                          # Current state at this node
        self.parent = parent                        # Parent node in the tree
        self.action = action                        # Action taken to reach this node
        self.children = []                          # Child nodes
        self.untried = [a for a in problem.actions(state)]  # Actions not yet tried from this state
        self.value = 0.0                            # Total accumulated reward
        self.visits = 0                             # Number of times this node was visited

    def __repr__(self):
        """
        Returns a debug-friendly string representation of the node.
        """
        return (f"MctsNode(action={self.action}, visits={self.visits}, "
                f"value={self.value:.2f}, untried={len(self.untried)})")

    def uct_val(self, c=EXPLORATION_CONSTANT):
        """
        Computes the Upper Confidence Bound (UCT) value for this node.
        """
        if self.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return (self.value / self.visits) + c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)

    def select(self, c=EXPLORATION_CONSTANT):
        """
        Traverses the tree to select the most promising node using UCT until an expandable node is found.
        """
        node = self
        while node.children and not node.untried:
            node = max(node.children, key=lambda child: child.uct_val(c))  # Select child with max UCT
        return node

    def expand(self):
        """
        Expands the tree by selecting an untried action and creating a new child node.
        """
        action = random.choice(self.untried)
        self.untried.remove(action)
        next_state = self.problem.enact(self.state, action)
        child = MctsNode(self.problem, next_state, self, action=action)
        self.children.append(child)
        return child

    def simulate(self):
        """
        Simulates a random playout from this node's state to a terminal state and returns the outcome.
        """
        prob = self.problem
        state = self.state
        player = prob.player(state)  # Player whose turn is next

        while not prob.isEnd(state):
            a = random.choice(prob.actions(state))  # Choose a random legal action
            state = prob.enact(state, a)            # Transition to next state

        return prob.reward(state) * (1 if player == "O" else -1)  # Score from "O"'s perspective

    def backpropagate(self, reward):
        """
        Propagates the result of a simulation back up the tree, updating value and visit counts.
        """
        node = self
        while node is not None:
            node.value += reward
            node.visits += 1
            reward = -reward  # Alternate reward sign due to alternating turns
            node = node.parent


# ------------ MCTS Core Algorithm ------------

def mcts(problem, state, episodes=EPISODES_PER_MOVE, c=EXPLORATION_CONSTANT):
    """
    Runs Monte Carlo Tree Search starting from the given state for a fixed number of episodes.
    Returns the root of the resulting search tree.
    """
    root = MctsNode(problem, state)
    for _ in range(episodes):
        node = root.select(c)
        if node.untried:
            node = node.expand()
        result = node.simulate()
        node.backpropagate(result)
    return root


# ------------ Move Selection Interface ------------

def choose_move(problem, state, episodes=EPISODES_PER_MOVE, c=EXPLORATION_CONSTANT):
    """
    Runs MCTS from the current state and returns the action with the highest visit count.
    """
    root = mcts(problem, state, episodes, c)
    best = max(root.children, key=lambda ch: ch.visits)
    return best.action
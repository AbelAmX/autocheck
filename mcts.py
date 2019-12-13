import numpy as np
import math
import collections
import checkers
import net

# Exploration constant c is defined as C_input
C_input = 1
# The Larges number of possible moves form at any point in the game
Large_move = 40


class ParentRootNode(object):
    def __init__(self):
        self.parent = None
        self.child_number_visits = collections.defaultdict(float)
        self.child_simulation_reward = collections.defaultdict(float)


class Node(object):
    def __init__(self, board, possible_moves, player, move=None, parent=None):
        self.board = board
        self.player = player
        self.is_expanded = False
        self.move = move   # index of the move that resulted in current Node/State
        self.possible_moves = possible_moves
        self.illegal_moves = Large_move - len(possible_moves)
        self.parent = parent
        self.child_prior_probability = np.zeros([Large_move], dtype=np.float32)
        self.child_number_visits = np.zeros([Large_move], dtype=np.float32)
        self.child_simulation_reward = np.zeros([Large_move], dtype=np.float32)
        self.children = {}

    def N(self):
        return self.parent.child_number_vists[self.move]

    def N(self, value):
        self.parent.child_number_vists[self.move] = value

    def R(self):
        return self.parent.child_simulation_reward[self.move]

    def R(self, value):
        self.parent.child_simulation_reward[self.move] = value

    def Q(self):
        return self.R() / (1 + self.N())

    def child_Q(self):
        return self.child_simulation_reward/(1 + self.child_number_visits)

    def child_U(self):
        return C_input * math.sqrt(self.N())*abs(self.child_prior_probability)/(1 + self.child_number_visits)

    def child_score(self):
        return self.child_Q() + self.child_U()

    def best_child(self):
        return np.argmax(self.child_number_visits + self.child_score()/100)

    def select_leaf(self):
        current = self
        while current.is_expanded:
            current = current.maybe_add_child(np.argmax(self.child_score()))
        return current

    def maybe_add_child(self, move):
        if move not in self.children:
            new_board = checkers.apply_move(self.board, self.possible_moves[move][0],
                                            self.possible_moves[move][1], self.player)
            player2 = checkers.switch_player(self.player)
            self.children[move] = Node(new_board, checkers.get_all_moves(new_board, player2),
                                       player2, move=move, parent=self)
        return self.children[move]

    def backpropagate(self, value):
        current_node = self
        while current_node.parent is not None:
            current_node.N += 1
            current_node.R += value
            value *= -1
            current_node = current_node.parent

    def expand_and_evaluate(self, child_pr):
        self.is_expanded = True
        for i in range(len(self.possible_moves), len(child_pr)):
            child_pr[i] = 0
        scale = child_pr.sum()
        self.child_prior_probability = child_pr/scale

    def print_tree(self):
        print("The Number of visits of next possible states")
        print(self.child_number_visits)
        print("The Reward of each next possible states")
        print(self.child_simulation_reward)
        print("The prior probability of next possible states")
        print(self.child_prior_probability)
        print("The possible moves already expanded")
        print(self.children)


def MCTS_Search_AI(board, player, num_reads, n_net):
    root = Node(board, checkers.get_all_moves(board, player), player, move=None, parent=ParentRootNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        child_prior_prob, value = n_net(checkers.get_state(leaf.board))
        print(child_prior_prob)
        if checkers.isTerminal(board):
            leaf.backpropagate(value)
        else:
            leaf.expand_and_evaluate(child_prior_prob)
            leaf.backpropagate(value)
        print("The %d number of reads", i)
        leaf.print_tree()
    return root

def MCTS_Search(board, player, num_reads, n_net):
    root = Node(board, checkers.get_all_moves(board, player), player, move=None, parent=ParentRootNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        child_prior_prob, value = n_net(checkers.get_state(leaf.board))
        print(child_prior_prob)
        if checkers.isTerminal(board):
            leaf.backpropagate(value)
        else:
            leaf.expand_and_evaluate(child_prior_prob)
            leaf.backpropagate(value)
        print("The %d number of reads", i)
        leaf.print_tree()
    return root


def policy(node, temp=1):
    return (node.child_number_visits**(1/temp))/sum(node.child_number_visits**(1/temp))


def MCTS_self_play():
    return


board_n = checkers.initial_board(8, 8)
player_n = 1
possible_moves_n = checkers.get_all_moves(board_n, player_n)

a = Node(board_n, possible_moves_n, player_n, ParentRootNode())


MCTS_Search(board_n, player_n, 10, net.ConvNet())



























import numpy as np
import math
import collections
import checkers
import checkers_Nnet as Nnet
import pickle
from tqdm import tqdm
import torch
import time
from torch.utils.data import Dataset

# Exploration constant c is defined as C_input
C_input = 1
# The Larges number of possible moves form at any point in the game
Large_move = 40
board_size = 8


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

    @property
    def N(self):
        return self.parent.child_number_visits[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def R(self):
        return self.parent.child_simulation_reward[self.move]

    @R.setter
    def R(self, value):
        self.parent.child_simulation_reward[self.move] = value

    @property
    def Q(self):
        return self.R() / (1 + self.N())

    def child_Q(self):
        return self.child_simulation_reward/(1 + self.child_number_visits)

    def child_U(self):
        return C_input * math.sqrt(self.N)*abs(self.child_prior_probability)/(1 + self.child_number_visits)

    def child_score(self):
        return self.child_Q() + self.child_U()

    def best_child(self):
        return np.argmax(self.child_number_visits + self.child_score()/100)

    def select_leaf(self):
        current = self
        while current.is_expanded:
            current = current.maybe_add_child(np.argmax(current.child_score()[0:(len(current.possible_moves))]))
        return current

    def maybe_add_child(self, move):
        # print(move)
        # print("Possible moves")
        # print(len(self.possible_moves))
        if len(self.possible_moves) == move:
            print(self.child_score())
            print(move)
            print("Possible moves")
            print(len(self.possible_moves))
            print(self.child_prior_probability)
        if move not in self.children:
            new_board = checkers.apply_move(self.board, self.possible_moves[move][0],
                                            self.possible_moves[move][1], self.player)
            player2 = checkers.switch_player(self.player)
            if self.is_board_in_MCTS(new_board, player2):
                m = self.child_score()
                m[move] = m.min()-1
                move = np.argmax(m[0:(len(self.possible_moves))])
                new_board = checkers.apply_move(self.board, self.possible_moves[move][0],
                                                self.possible_moves[move][1], self.player)
                player2 = checkers.switch_player(self.player)
            self.children[move] = Node(new_board, checkers.get_all_moves(new_board, player2),
                                       player2, move=move, parent=self)
        #checkers.print_board(self.children[move].board)
        return self.children[move]

    def is_board_in_MCTS(self, board, player):
        current = self
        while current.parent is not None:
            if np.array_equal(current.board, board) and current.player == player:
                return True
            current = current.parent
        return False

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


# def MCTS_Search_AI(board, player, num_reads, n_net):
#     root = Node(board, checkers.get_all_moves(board, player), player, move=None, parent=ParentRootNode())
#     for i in range(num_reads):
#         leaf = root.select_leaf()
#         child_prior_prob, value = n_net(checkers.get_state(leaf.board))
#         print(child_prior_prob)
#         if checkers.isTerminal(board):
#             leaf.backpropagate(value)
#         else:
#             leaf.expand_and_evaluate(child_prior_prob)
#             leaf.backpropagate(value)
#         print("The %d number of reads", i)
#         leaf.print_tree()
#     return root

def MCTS_Search(board, player, num_reads, n_net):
    root = Node(board, checkers.get_all_moves(board, player), player, move=None, parent=ParentRootNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        player = checkers.switch_player(player)
        child_prior_prob, value = n_net(torch.FloatTensor(checkers.get_state2(leaf.board, leaf.player)))
        # print(child_prior_prob)
        # print("The number of reads", i)
        if checkers.isTerminal(board) or checkers.get_all_moves(leaf.board, leaf.player) == []:
            # print("Finished Game")
            leaf.backpropagate(value)
            # leaf.print_tree()
        else:
            child_prior_prob = child_prior_prob.cpu().detach().numpy().reshape(-1)
            leaf.expand_and_evaluate(child_prior_prob)
            leaf.backpropagate(value)

    # root.print_tree()
    return root


def get_policy(node, temp=1):
    return (node.child_number_visits**(1/temp))/sum(node.child_number_visits**(1/temp))


def MCTS_self_play(nnet, num_games, s_index, iteration):
    data_x = []
    for itt in tqdm(range(s_index, num_games + s_index)):
        board = checkers.initial_board(board_size, board_size)
        player = 1
        data = []
        value = 0
        num_moves = 0
        t = 1
        while checkers.isTerminal(board) is not True:
            if num_moves > 15:
                t = 0.1
            root = MCTS_Search(board, player, 200, nnet)
            # print("The turn of player {:d} and Moves {:d}".format(player, num_moves))
            # checkers.print_board(root.board)
            policy = get_policy(root, t)
            data.append([board, player, policy])
            move = np.argmax(policy)
            board = checkers.apply_move(root.board, root.possible_moves[move][0], root.possible_moves[move][1],
                                        root.player)
            player = checkers.switch_player(player)
            if len(checkers.get_all_moves(board, player)) == 0:
                # Player == 1 means White pieces
                print("Game Finished")
                if player == 1:
                    value = 1
                elif player == 2:
                    value = -1
                else:
                    value = 0
                break
            if num_moves == 150:
                value = 0
                break
            num_moves += 1

        for ind, dx in enumerate(data):
            s, pl, po = dx
            if ind == 0:
                data_x.append([checkers.get_state2(s, pl), po, 0])
            else:
                data_x.append([checkers.get_state2(s, pl), po, value])
        del data
        # filename = "MCTS_iteration-{:d}_game-{:d}.p".format(iteration, itt)
        # save_data(filename, data_x)
    return data_x


def training(n_net, batch_size, n_epochs, learning_rate, dataset, iteration):
    n_net.train()
    criteria = Nnet.ErrorFnc()
    train_set = TrainingData(dataset)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(n_net.parameters(), lr=learning_rate, betas=(0.8, 0.999))
    t_time = time.time()

    for epoch in range(n_epochs):
        running_loss = 0
        total_loss = 0
        start_time = time.time()

        for i, data in enumerate(train_loader, 0):
            state, policy, value = data

            policy_estimate, value_estimate = n_net(state.float())
            loss = criteria(value.float(), value_estimate, policy.float(), policy_estimate)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            total_loss += loss.item()

            # if i % 10 == 0:
            #     print("Epoch-{:d}_training_loss-{:.2f}_took-{:.2f}s".format(epoch+1,
            #                                                                 running_loss, time.time() - start_time))
            #     start_time = time.time()
            #     running_loss = 0

        print("Training Finished, iteration-{:d} total loss-{:.2f} and took-{:.2f}s".format(iteration, total_loss, time.time() - t_time))


def save_data(name, data):
    data1 = open(name, 'wb')
    pickle.dump(data, data1, protocol=pickle.HIGHEST_PROTOCOL)
    data1.close()


def load_data(name):
    c_name = name + ".p"
    data1 = open(c_name, 'rb')
    return pickle.load(data1)


class TrainingData(Dataset):
    def __init__(self, data_set):
        d = np.array(data_set)
        self.a = d[:, 0]
        self.b = d[:, 1]
        self.c = d[:, 2]

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return np.int64(self.a[i]), self.b[i], self.c[i]

# board_n = checkers.initial_board(8, 8)
# player_n = 1
# possible_moves_n = checkers.get_all_moves(board_n, player_n)
# print(possible_moves_n)
# a = Node(board_n, possible_moves_n, player_n, ParentRootNode())
# MCTS_Search(board_n, player_n, 2000, Nnet.Net())


# d = MCTS_self_play(Nnet.Net(), 1, 0, 1)
# training(Nnet.Net(), 1, 50, 0.001, d)

def evaluate(n_net, n_games):
    iteration = 10
    s_index = 0
    for i in range(iteration):
        data = MCTS_self_play(n_net.Net(), n_games, 0, 1)
        training(n_net.Net(), 1, 50, 0.001, data, i)


evaluate(Nnet, 1)


















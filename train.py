### FULL CONNECT FOUR ALPHAZERO IMPLEMENTATION ###
### HYPERPARAMETERS MAY NEED TUNING AND ARE AT THE BOTTOM OF THE FILE ###

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import random
from tqdm.notebook import trange
import copy
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConnectFour:
  def __init__(self):
    self.row_count = 6
    self.column_count = 7
    self.action_size = self.column_count
    self.in_a_row = 4
  def __repr__(self):
    return "Connect4"

  def get_initial_state(self):
    return np.zeros((self.row_count, self.column_count))

  def get_next_state(self, state, action, player):
    row = np.max(np.where(state[:, action] == 0)) #finds row with highest number
    column = action
    state = state.copy()
    state[row, column] = player
    return state

  def get_valid_moves(self, state):
    return (state[0] == 0).astype(np.uint8)

  def check_win(self, state, action):
    # returns if board is won
      
        if action == None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        player = state[row][action]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )

  def get_value_and_terminated(self, state, action):
    if self.check_win(state, action):
      return 1, True
    if np.sum(self.get_valid_moves(state)) == 0:
      return 0, True
    return 0, False

  def get_player(self, state):
      player = 1 if np.sum(state) == 0 else -1
      return player
      
  def get_opponent(self, player):
    return -player

  def get_opponent_value(self, value):
    return -value

  def change_perspective(self, state, player):
    return state * player

  def get_encoded_state(self, state):
    # turns state into encoded version for model
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)

    if len(state.shape) == 3: #check if state has batch axis, normally theres 2
      encoded_state = np.swapaxes(encoded_state, 0, 1) #swap 0th and 1st axis
    return encoded_state
      
  def augment_data(self, data):
    states, probs, values = zip(*data)
    for state, prob, value in zip(states, probs, values):
        data.append((
            np.fliplr(state),
            prob[::-1],
            value
        ))

class ResNet(nn.Module):
  def __init__(self, game, num_resBlocks, num_hidden, device):
    super().__init__()

    self.device = device
    self.startBlock = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=num_hidden, kernel_size=3, padding=1), #in_channels=3
        nn.BatchNorm2d(num_features=num_hidden),
        nn.ReLU(),
    )

    self.backBone = nn.ModuleList(
        [ResBlock(num_hidden) for i in range(num_resBlocks)]
    )

    self.policyHead = nn.Sequential(
        nn.Conv2d(in_channels=num_hidden, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=32*game.row_count*game.column_count, out_features=game.action_size)
    )

    self.valueHead = nn.Sequential(
        nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3*game.row_count*game.column_count, 1),
        nn.Tanh()
    )
    self.to(device)

  def forward(self, x):
    x = self.startBlock(x)
    for resBlock in self.backBone:
      x = resBlock(x)
    policy = self.policyHead(x)
    value = self.valueHead(x)
    return policy, value

class ResBlock(nn.Module):
  def __init__(self, num_hidden):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(num_hidden)
    self.conv2 = nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(num_hidden)

  def forward(self, x):
    residual = x #apparently better to sum x to x after layers
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = x + residual
    x = F.relu(x)
    return x

class Node:
  def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
    self.game = game
    self.args = args
    self.state = state
    self.parent = parent
    self.action_taken = action_taken
    self.prior = prior

    self.children = []

    self.visit_count = visit_count
    self.value_sum = 0

  def is_fully_expanded(self):
    return len(self.children) > 0
    #np.sum turns True to 1 and False to 0, checking if all moves are taken

  def select(self):
    # returns node with highest ucb score
    best_child = None
    best_ucb = -np.inf

    for child in self.children:
      ucb = self.get_ucb(child)
      if ucb > best_ucb:
        best_child = child
        best_ucb = ucb

    return best_child

  def get_ucb(self, child):
    # returns ucb value for a node
    if child.visit_count == 0:
      q_value = 0
    else:
      #q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2  #want lowest score for opponent
      q_value = -(child.value_sum) / child.visit_count
    return q_value + self.args["C"] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

  def expand(self, policy):
    # expands the node with given policy
      
    for action, prob in enumerate(policy):
      if prob > 0:
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1) #player plays as if going first 
        child_state = self.game.change_perspective(child_state, player = -1)

        child = Node(self.game, self.args, child_state, self, action, prob)
        self.children.append(child)

  def backpropogate(self, value):
      self.value_sum += value
      self.visit_count += 1

      value = self.game.get_opponent_value(value) #change since previous player is opponent
      if self.parent is not None:
        self.parent.backpropogate(value) #recursion

class MCTSParallel:
  def __init__(self, game, args, model, searches):
    self.game = game
    self.args = args
    self.model = model
    self.searches = searches
  @torch.inference_mode()
  def search(self, states, spGames, use_dirichlet=True):
    #define root
    policy, _ = self.model(
        torch.tensor(self.game.get_encoded_state(states), device = self.model.device)
    )
    policy = torch.softmax(policy, dim=1).cpu().numpy()

    #dirichlet noise
    if use_dirichlet:
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] \
              * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size, size=policy.shape[0])

    for i, spg in enumerate(spGames):
      spg_policy = policy[i]
      valid_moves = self.game.get_valid_moves(states[i])
      spg_policy *= valid_moves
      spg_policy /= np.sum(spg_policy)
      # store Node in spg root
      spg.root = Node(self.game, self.args, states[i], visit_count=1) 
      #set visit count to 1 as its immediately expanded
      spg.root.expand(spg_policy)
    
    for search in range(self.searches):
      for spg in spGames:
        spg.node = None
        node = spg.root

        while node.is_fully_expanded():
          node = node.select() # find best child based on UCB, SELECTION

        value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
        value = self.game.get_opponent_value(value) #node contains action taken by opponent, so value must be switched

        if is_terminal:
          node.backpropogate(value)

        else:
          spg.node = node

      #get all expandable games, not terminal
      expandable_spGames = [mappingIdx for mappingIdx in range((len(spGames))) if spGames[mappingIdx].node is not None]
      if len(expandable_spGames) > 0:
        states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
        policy, value = self.model(
            torch.tensor(self.game.get_encoded_state(states), device = self.model.device)
        )
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        value = value.cpu().numpy()

      for i, mappingIdx in enumerate(expandable_spGames):
        node = spGames[mappingIdx].node
        spg_policy = policy[i]
        spg_value = value[i]

        valid_moves = self.game.get_valid_moves(node.state)
        spg_policy *= valid_moves
        spg_policy /= np.sum(spg_policy)

        node.expand(spg_policy)
        node.backpropogate(spg_value)

class AlphaZeroParallel:
  def __init__(self, model, optimizer, game, args):
    self.model = model
    self.optimizer = optimizer
    self.game = game
    self.args = args
    self.mcts = MCTSParallel(game, args, model, searches=self.args["num_searches"])
    self.previous_model = copy.deepcopy(self.model.state_dict())

    total_steps = args["num_iterations"] * args["num_epochs"] 
    self.scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps, 
        eta_min=1e-6 # Set a minimum learning rate floor
    )

  def selfPlay(self):
    # self play loop, returns training data
    return_memory = []
    player = 1
    spGames = [SPG(self.game) for spg in range(self.args["num_parallel_games"])]
    move_number = 0
    while len(spGames) > 0:

      #get states
      states = np.stack([spg.state for spg in spGames])
      neutral_states = self.game.change_perspective(states, player = player) #function works with multiple states

      self.mcts.search(neutral_states, spGames)

      for i in range(len(spGames))[::-1]: #flip range for deleting
        spg = spGames[i]

        action_probs = np.zeros(self.game.action_size)
        for child in spg.root.children:
          action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs) # turn into probability

        spg.memory.append((spg.root.state, action_probs, player))

        if spg.move_count < self.args["temperature_cutoff"]:
            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
        else:
            action = np.argmax(action_probs)

        spg.state = self.game.get_next_state(spg.state, action, player)
        spg.move_count += 1
        value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

        if is_terminal:
          for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
            hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
            return_memory.append((
              self.game.get_encoded_state(hist_neutral_state),
              hist_action_probs,
              hist_outcome
            ))

          del spGames[i]
      move_number += 1
      player = self.game.get_opponent(player)
    return return_memory
  def testPlay(self, self_going_first):
    total_move_count = 0
    winrate = 0
      
    opponent = ResNet(self.game, 9, 128, self.model.device)
    opponent.load_state_dict(self.previous_model)
    opponent.eval()
      
    mcts_opponent = MCTSParallel(self.game, self.args, opponent, self.args["num_test_searches"])

    player = 1 if self_going_first else -1
    spGames = [SPG(self.game) for spg in range(self.args["num_test_games"])]

    while len(spGames) > 0:
      #print(f"next iteration, sp games left {len(spGames)}")
      #get states
      states = np.stack([spg.state for spg in spGames])
      neutral_states = self.game.change_perspective(states, player = player) #function works with multiple states
      if player == 1:
          self.mcts.search(neutral_states, spGames)

      else:
          mcts_opponent.search(neutral_states, spGames)
      
      for i in range(len(spGames))[::-1]: #flip range for deleting
        spg = spGames[i]

        action_probs = np.zeros(self.game.action_size)
        for child in spg.root.children:
          action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs) # turn into probability

        spg.memory.append((spg.root.state, action_probs, player))
    
        action = np.argmax(action_probs)

        spg.state = self.game.get_next_state(spg.state, action, player)
        spg.move_count += 1
        value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

        if is_terminal:
          total_move_count += spg.move_count
          if player == value: #only increment if the tested model wins
              winrate += 1
          elif value == 0:
              winrate += 0.5
          del spGames[i]
      
      player = self.game.get_opponent(player)
    return total_move_count, winrate/self.args['num_test_games']
  def train(self, memory):
    random.shuffle(memory)
  
    for batchIdx in range(0, len(memory), self.args["batch_size"]):
      sample = memory[batchIdx:min(len(memory), batchIdx + self.args["batch_size"])] #never get out of bounds
      state, policy_targets, value_targets = zip(*sample) #turn tuples into lists
      state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
      #values must be in its own subbarray
      #convert to tensor
      state = torch.tensor(state, dtype=torch.float32, device = self.model.device)
      policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device = self.model.device)
      value_targets = torch.tensor(value_targets, dtype = torch.float32, device = self.model.device)

      out_policy, out_value = self.model(state)

      policy_loss = F.cross_entropy(out_policy, policy_targets)
      value_loss = F.mse_loss(out_value, value_targets)

      loss = policy_loss + value_loss
      #print(f"loss is {loss}")
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      #print(loss.item())

  def learn(self):
    for iteration in trange(self.args["num_iterations"]):
      memory = []

      self.model.eval()
      for selfPlay_iteration in trange(self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]):
        memory += self.selfPlay()

      print(f"memory length is {len(memory)} avg_moves: {(len(memory)/self.args['num_selfPlay_iterations']):.2f}")
           
      #game.augment_data(memory)
      print(f"Augmented Memory Length: {len(memory)}")
      self.model.train()
        
      for epoch in range(self.args["num_epochs"]):
        self.train(memory)
        self.scheduler.step()

      print("Testing against Previous Model...")
      self.model.eval()
        
      tmc_f, wr_f = self.testPlay(True) #model going first
      tmc_s, wr_s = self.testPlay(False) #model going first
      avg_wr = ((wr_f + wr_s)/2)
      
      avg_moves = (tmc_f + tmc_s) / self.args['num_test_games'] / 2

      print(f"avg_wr: {avg_wr:.2f} | first: {wr_f:.2f} | second: {wr_s:.2f} | avg_moves:{avg_moves:.2f}")

      # if avg_wr > 0.55:
      #     print("promoting current model to best model")
      self.previous_model = copy.deepcopy(self.model.state_dict())
          
      torch.save(self.model.state_dict(), f=f"model_{iteration}.pt")
      torch.save(self.optimizer.state_dict(), f=f"optimizer_{iteration}.pt")

      #self.args['temperature'] = max(0.5, self.args['temperature'] * 0.95)

class SPG:
  def __init__(self, game):
    self.state = game.get_initial_state()
    self.memory = []
    self.move_count = 0
    self.root = None
    self.node = None

game = ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
#model.load_state_dict(torch.load("/kaggle/input/nov25/pytorch/default/1/model_0.pt"))
optimizer = torch.optim.AdamW(params = model.parameters(), lr = 3e-4, weight_decay = 0.0001)
#optimizer.load_state_dict(torch.load("/kaggle/input/nov25/pytorch/default/1/optimizer_0.pt"))
args = {
    "C" :2, # exploration constant
    "num_searches": 600, # mcts searches per move
    "num_test_searches": 300, # mcts searches per move during testing
    "num_iterations": 8, # training iterations
    "num_selfPlay_iterations": 500, # self play games per training iteration
    "num_parallel_games": 250, # num of parallel self play games
    "num_epochs": 4, # number of training epochs per iteration
    "num_test_games": 50, # number of test games per iteration
    "batch_size": 256, # training batch size
    "temperature": 1.25, # initial temperature for exploration
    "temperature_cutoff": 10, # moves before temperature is not used
    "dirichlet_epsilon": 0.25, # random noise
    "dirichlet_alpha" : 0.3 # random noise
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()

import numpy as np

class ConnectFour:
  def __init__(self):
    self.row_count = 6
    self.column_count = 7
    self.action_size = self.column_count
    self.in_a_row = 4
    self.board = self.get_initial_state()
    self.current_player = 1
    self.last_move = None

  def __repr__(self):
    return "Connect4"

  def play_move(self, action):
    row = np.max(np.where(self.board[:, action] == 0)) #finds row with highest number
    column = action
    self.board[row, column] = self.current_player
    self.current_player = -self.current_player
    self.last_move = (row, column)

  def is_terminal(self):
    if self.last_move is None:
      return False
    _, is_terminal = self.get_value_and_terminated(self.board, self.last_move[1])
    return is_terminal

  def get_initial_state(self):
    return np.zeros((self.row_count, self.column_count))

  def get_next_state(self, state, action, player):
    row = np.max(np.where(state[:, action] == 0)) #finds row with highest number
    column = action
    state[row, column] = player
    return state

  def get_valid_moves(self, state):
    return (state[0] == 0).astype(np.uint8)

  def check_win(self, state, action):
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

  def get_opponent(self, player):
    return -player

  def get_opponent_value(self, value):
    return -value

  def change_perspective(self, state, player):
    return state * player

  def get_player(self, state):
    player = 1 if np.sum(state) == 0 else -1
    return player
    
  def get_encoded_state(self, state):
    # get planes of 1s or -1s denoting turn player
    # turn_plane = self.get_turn_planes(state)
      
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)

    if len(state.shape) == 3: #check if state has batch axis, normally theres 2
      encoded_state = np.swapaxes(encoded_state, 0, 1) #swap 0th and 1st axis
    return encoded_state
    
  # def get_turn_planes(self, state):
  #   turn_planes = []
  #   if len(state.shape) == 2:
  #       turn_plane = np.ones((self.row_count, self.column_count))
  #       if self.get_player(state) != 1:
  #           turn_plane *= -1 
  #       return turn_plane
  #   else:
  #       for s in state:
  #           turn_plane = np.ones((self.row_count, self.column_count))
  #           if self.get_player(s) != 1:
  #               turn_plane *= -1
  #           turn_planes.append(turn_plane)
  #   return np.stack(turn_planes, axis = 0)

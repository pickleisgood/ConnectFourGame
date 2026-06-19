# Connect4 AI (MCTS + ResNet + Pygame)

This project is an interactive **Connect 4 game with a Monte Carlo Tree Search (MCTS) AI powered by a PyTorch ResNet model**. The game is rendered in real time using **Pygame**, and the AI uses threaded asynchronous search so the UI remains fully responsive.

---

## Features

- Human vs AI Connect 4
- Real-time Pygame interface
- Deep learning (PyTorch) ResNet model for policy/value prediction
- Monte Carlo Tree Search for move selection
- Threaded AI computation (AI "thinks" without freezing the UI)
- Choose to play as Red or Yellow
- Includes saved model: `Connect4.pt`

---

## AI Architecture

The AI is composed of:
- **A ResNet** that predicts move probabilities and expected win value  
- **MCTS**, which uses the model for guided search  
- **UCB-based exploration**, Dirichlet noise, and configurable search count  

The model was trained through self-play and reinforcement learning.

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/pickleisgood/ConnectFourGame.git
cd ConnectFourGame
```

### 2. Install Requirements
```bash
pip3 install -r requirements.txt
```
### 3. Run the Program
```bash
python3 connect4_runner.py
```

---

## How to Train

If you want to train the model from scratch or continue training:

1. Open `train.py` and adjust the hyperparameters at the bottom of the file (e.g., `num_iterations`, `num_selfPlay_iterations`, `num_searches`) to suit your hardware.
2. Run the training script:
```bash
python3 train.py
```
3. The script will generate self-play games, train the neural network, and test it against the previous iteration. Models will be saved as `model_*.pt` at each iteration.

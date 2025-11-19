import pygame
import time
import sys
import numpy as np
import torch
from connect4 import ConnectFour
from ResNet_Connect4 import ResNet, ResBlock
from CONNECT4mcts import MCTS
import concurrent.futures

device = "cpu"

#display
pygame.init()
size = width, height = 800, 800
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Connect 4")
clock = pygame.time.Clock() # create object to control framerate

# Use system fonts instead of custom font file
mediumFont = pygame.font.SysFont('arial', 28)
largeFont = pygame.font.SysFont('arial', 40)
moveFont = pygame.font.SysFont('arial', 60)

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
yellow = (255, 255, 0)

args = {
    "C" : 1, #Exploration constant
    "num_searches": 800, #Adjust to increase/decrease strength and speed
    "dirichlet_epsilon": 0.25, #Random Noise
    "dirichlet_alpha" : 0.3 #Random Noise
}

def get_ai_move(game, mcts):
    neutral_state = game.change_perspective(game.board, game.current_player) # player is -1 in this case
    probs, value = mcts.search(neutral_state)
    action = np.argmax(probs) 
    print(f"probs: {probs}, value: {value:.4f}")
    return action

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
ai_threading = False
    
def run():
    #initialize game
    game = ConnectFour()
    human_player = None
    model = ResNet(game, 7, 128, 0.2, device)
    
    model.load_state_dict(torch.load("Connect4.pt", map_location=device))
    model.eval()
    global ai_threading

    while True:
        mcts = MCTS(game, args, model)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        screen.fill(black)
        
        if human_player is not None:
            tile_size = 80
            tile_origin = ((width - (7 * tile_size))/2,
                          (height - (6 * tile_size))/2)
            tiles = []
            
            # Draw the board
            for i in range(6):
                row = []
                for j in range(7):
                    rect = pygame.Rect(
                        tile_origin[0] + j * tile_size,
                        tile_origin[1] + i * tile_size,
                        tile_size, tile_size
                    )
                    pygame.draw.rect(screen, white, rect, 3)
                    
                    # Draw the pieces
                    if game.board[i][j] != 0:
                        color = red if game.board[i][j] == 1 else yellow
                        pygame.draw.circle(
                            screen,
                            color,
                            (rect.centerx, rect.centery),
                            tile_size // 2 - 5
                        )
                    row.append(rect)
                tiles.append(row)
            
            # Handle game state
            if not game.is_terminal():
                current_player = game.current_player
                
                if current_player != human_player:
                    title = largeFont.render("AI THINKING...", True, white)
                    
                    if not ai_threading:
                        ai_threading = True
                        thread = executor.submit(get_ai_move, game, mcts)
                    
                    if thread.done():
                        action = thread.result()
                        ai_threading = False

                        game.play_move(action)

                else:
                    title = largeFont.render("YOUR TURN", True, white)
                    click, _, _ = pygame.mouse.get_pressed()
                    if click == 1:
                        mouse = pygame.mouse.get_pos()
                        for i in range(6):
                            for j in range(7):
                                if tiles[i][j].collidepoint(mouse):
                                    if game.board[0][j] == 0:
                                        game.play_move(j)
                                        time.sleep(0.2)  # Prevent multiple clicks
            else:
                value, _ = game.get_value_and_terminated(game.board, game.last_move[1])
                if value == 0:
                    title = largeFont.render("Game Over: Tie", True, white)
                else:
                    title = largeFont.render(f"Winner is {-game.current_player}", True, white)
                
                againButton = pygame.Rect(width / 3, height - 65, width / 3, 50)
                again = mediumFont.render("Play Again", True, black)
                againRect = again.get_rect()
                againRect.center = againButton.center
                pygame.draw.rect(screen, white, againButton)
                screen.blit(again, againRect)
                
                click, _, _ = pygame.mouse.get_pressed()
                if click == 1:
                    mouse = pygame.mouse.get_pos()
                    if againButton.collidepoint(mouse):
                        time.sleep(0.2)
                        game = ConnectFour()
                        human_player = None
        else:
            title = largeFont.render("Play Connect 4", True, white)
            titleRect = title.get_rect()
            titleRect.center = ((width / 2), 50)
            screen.blit(title, titleRect)
            
            playXButton = pygame.Rect((width / 8), (height / 2), width / 4, 50)
            playX = mediumFont.render("Play as Red", True, black)
            playXRect = playX.get_rect()
            playXRect.center = playXButton.center
            pygame.draw.rect(screen, white, playXButton)
            screen.blit(playX, playXRect)
            
            playOButton = pygame.Rect(5 * (width / 8), (height / 2), width / 4, 50)
            playO = mediumFont.render("Play as Yellow", True, black)
            playORect = playO.get_rect()
            playORect.center = playOButton.center
            pygame.draw.rect(screen, white, playOButton)
            screen.blit(playO, playORect)
            
            click, _, _ = pygame.mouse.get_pressed()
            if click == 1:
                mouse = pygame.mouse.get_pos()
                if playXButton.collidepoint(mouse):
                    time.sleep(0.2)
                    human_player = 1
                elif playOButton.collidepoint(mouse):
                    time.sleep(0.2)
                    human_player = -1
        
        titleRect = title.get_rect()
        titleRect.center = ((width / 2), 50)
        screen.blit(title, titleRect)
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    run()
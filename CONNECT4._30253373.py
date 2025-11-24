# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt 
import random
    
board = np.zeros((6,7)) #6x7 Matrix of zeros
print(board)
game_finished = False 

print("Player 1=1(red), Computer=2(yellow). Player 1 starts.")



def bot_input(): #This corresponds to the Computer move  
    return random.randint(0,6)
    
def place_counter(cn, board, player): 
    for i in range(6):
        if (board[5-i, cn] == 0):
            board[5-i, cn]=player
            return board, 5-i
            break
    return board, 0
        
def check_horizontal_for_win(board): 
   for j in range(6):
       for k in range(4):
           if (board[j, k] == 1 and board[j, k+1] == 1 and board[j, k+2] == 1 and board[j, k+3] == 1):
               print("You Win!")
               return True
           elif (board[j, k] == 2 and board[j, k+1] == 2 and board[j, k+2] == 2 and board[j, k+3] == 2): 
               print("You Lose.")
               return True
   return False

def check_vertical_for_win(board):
   for j in range(3):
       for k in range(7):
           if (board[j, k] == 1 and board[j+1, k] == 1 and board[j+2, k] == 1 and board[j+3, k] == 1):
               print("You Win!")
               return True
           elif (board[j, k] == 2 and board[j+1, k] == 2 and board[j+2, k] == 2 and board[j+3, k] == 2):
               print("You Lose.")
               return True
   return False
           
             

def check_diagonal_for_win(board):
    # Positive Slope
    for j in range(3):
       for k in range(4):
           if (board[j, k] == 1 and board[j+1, k+1] == 1 and board[j+2, k+2] == 1 and board[j+3, k+3] == 1):
               print("You Win!")
               return True
           elif (board[j, k] == 2 and board[j+1, k+1] == 2 and board[j+2, k+2] == 2 and board[j+3, k+3] == 2):
               print("You Lose.")
               return True
           #Negative Slope
    for j in range(3):
       for k in range(3,7):
           if (board[j, k] == 1 and board[j+1, k-1] == 1 and board[j+2, k-2] == 1 and board[j+3, k-3] == 1):
               print("You Win!")
               return True
           elif (board[j, k] == 2 and board[j+1, k-1] == 2 and board[j+2, k-2] == 2 and board[j+3, k-3] == 2):
               print("You Lose.")
               return True
    return False

def check_for_draw(board):
    if np.count_nonzero(board) == 42:
        print("It's A Draw!")
        return True
    return False

def block_vertical_for_three_in_a_row(board):
#Stops Player 1 from getting vertical wins
   for j in range(3):
       for k in range(7):
           if (board[5-j, k] == 1 and board[5-(j+1), k] == 1 and board[5-(j+2), k] == 1):
               if board[5-(j+3), k] == 0:
                   return True, k
   return False, ""

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('blue')
ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_title('Connect 4 Board')
ax.set_xlim([-1, 7])
ax.set_ylim([-1, 6])
plt.xticks([0,1,2,3,4,5,6])
plt.yticks([0,1,2,3,4,5])

            
while not game_finished:

    P1 = 1
    P2 = 2
    valid_entry = False
    while not valid_entry:
        cnstring = (input("To drop a piece, enter number from 0(leftmost column) to 6(rightmost column)"))
        try: 
            cn = int(cnstring)
            while cn > 6 or cn < 0 :
                print("Invalid entry. Please enter number between 0 and 6.")
                cn = int(input("To drop a piece, enter number from 0(leftmost column) to 6(rightmost column)"))
            valid_entry = True
        except: print("Invalid entry. Please enter number between 0 and 6.")
        
    
    
    
    board, rn = place_counter(cn, board, P1)
    game_finished = check_for_draw(board) or check_horizontal_for_win(board) or check_vertical_for_win(board) or check_diagonal_for_win(board)
    ax.plot(cn, 5-rn, color='red', marker='o', markersize=20, linestyle="None")
    if not game_finished:
        a,b = block_vertical_for_three_in_a_row(board)
        if a == True:
            cn = b
        elif a == False:    
            cn = bot_input()  
        board, rn = place_counter(cn, board, P2)
        print(board)
        game_finished = check_for_draw(board) or check_horizontal_for_win(board) or check_vertical_for_win(board) or check_diagonal_for_win(board)
        ax.plot(cn, 5-rn, color='yellow', marker='o', markersize=20, linestyle="None")
plt.show()

                    

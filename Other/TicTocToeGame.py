# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 15:09:45 2021

@author: golan
"""

from IPython.display import clear_output
import random


row_num = 3
board = [0]*9
free_indices = len(board)
players = [["",""], ["",""]]
first_game = True

#tic tac toe game

def isPositionFree(pos):
    idx = int(pos) -1
    return board[idx] not in ("X", "O")    

# this function is called only when free_indices > 0
def getUserPosition(num):
    global free_indices
    
    result = input(f"{players[num][0]} please pick a free position from 1 to {len(board)} ")
    while True:
        if result.isdigit() == False:
            print("you must put in a number!")
        elif int(result) not in range(1,len(board)+1):
            print(f"the number must be from 1-{len(board)}")
        elif isPositionFree(result) == False:
            print("the position you selected is already occupied")
        else:
            free_indices -= 1
            break 
        displayBoard()
        result = input(f"{players[num][0]} please pick a free position from 1 to {len(board)} ")
    return result

def initUser():
    global first_game
    
    if first_game == True:
        name = input("player 1, please enter your name: ")
    
        xo_char = input("Do you want to play X or O? ")    
        while xo_char.lower() not in ("x", "o"):
            print("I'm sorry that is not a valid choice") 
            xo_char = input("Please write x if you want to be X or o otherwise ") 
        players[0] = [name, xo_char.upper()]
    
        name = input("player 2, please enter your name: ")
        if players[0][1] == "X":
            xo_char = "O"
        else:
            xo_char = "X"
        players[1] = [name, xo_char.upper()]
        first_game = False
    
    return random.randint(0, 1) 

def initBoard():
    global free_indices
    global board
    
    free_indices = row_num*row_num
    board = [0]*(row_num*row_num)
    
    for i in range(len(board)-row_num+1):
        for j in range(row_num):
            board[i+j] = str(i+j+1)

def displayBoard():
    print_row = ''
    row_idx = 0
    clear_output()
    
    while row_idx < len(board):
        print_row = "\t "
        for val in range(row_idx, row_idx+row_num):
            if len(board[val]) == 1:
                print_row += "  " + board[val] + "  |"
            else:
                print_row += "  " + board[val] + " |"
        #we have our row    
        spaces = "     |"
        lines = "_____|"
        
        #print("\t     |     |")
        print("\t", spaces*row_num)
        print(print_row)
        #print('\t_____|_____|_____')
        print('\t', lines*row_num)
        print_row = ""
        row_idx += row_num

def isGameWon():
    #check rows
    row_idx = 0
    while row_idx < len(board):
        row_strick = 1
        for i in range(row_idx, row_idx+row_num-1):
            if board[i] == board[i+1]:
                row_strick +=1
            else:
                break
        if row_strick == row_num:
            return True
        row_idx += row_num
        
    #check columns
    for col_idx in range(row_num):
        col_strick = 1
        col = col_idx
        while col in range(len(board)-row_num):
            if board[col] == board[col + row_num]:
                col_strick +=1
                col += row_num
            else:
                break
        if col_strick == row_num:
            return True
    
    #check diagonals
    diag_idx = 0
    diag_strick = 1
    while diag_idx < len(board)-(row_num+1):
        if board[diag_idx] == board[diag_idx+row_num+1]:
            diag_strick +=1
        else:
            break
        diag_idx += row_num+1
    if diag_strick == row_num:
        return True
        
    diag_idx = row_num - 1
    diag_strick = 1
    while diag_idx < len(board)-(row_num+1):
        if board[diag_idx] == board[diag_idx+row_num-1]:
            diag_strick +=1
        else:
            break
        diag_idx += row_num-1
    if diag_strick == row_num:
            return True
    return False

def getRowNum():
    global row_num
    rows = input("how many rows? ")
    while rows.isdigit() == False or int(rows) < 3 or int(rows) > 15:
        print("please pick a number between 3 and 15")
        rows = input("how many rows? ")
        if rows.lower() == "exit" or rows.lower() == "quit" or rows.lower() == "stop" or rows.lower() == "no":
            return False
            break
    row_num = int(rows)
    return True
    

def ticTacToeGame():
    print("Great! let's start")
    ret = getRowNum()
    if ret == False:
        return
    else:
        initBoard()  
        turn = initUser()
        displayBoard()
        
        while free_indices > 0:
            result = getUserPosition(turn) #returns a string digit 1-9 that is free in the board
            idx = int(result) - 1
            board[idx] = players[turn][1]
            displayBoard()
            if isGameWon():
                print(f"Game over! {players[turn][0]} is the winner!")
                break
            elif free_indices == 0:
                print("Game over! it's a tie")
                break
            else:
                if turn == 0:
                    turn = 1
                else:
                    turn = 0

    
def ticTacToe():
    print("welcome to the tic tac toe game!")
    answer = input("Do you want to play? ")
    while answer.lower() == "yes" or answer.lower() == "yea" or answer.lower() == "y":
        ticTacToeGame()
        answer = input("one more? ")
    print("see you next time! :)")

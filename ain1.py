import copy
import numpy as np
import random
import math
import tensorflow as tf
import keras
colLen = 7
rowLen = 6
posEmpty = 0
import collections

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(colLen-3):
        for r in range(rowLen):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(colLen):
        for r in range(rowLen-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(colLen-3):
        for r in range(rowLen-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(colLen-3):
        for r in range(3, rowLen):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def get_valid_columns(board):
    valid_columns = []
    for c in range(colLen):
        if (board[rowLen - 1][c] == posEmpty):
            valid_columns.append(c)

    return valid_columns
def drop_board(board, col, player):
    for r in range(rowLen):
        if board[r][col] == 0:
            board[r][col] = player
            break
    return board
def get_reward(board, player, col):
    # checking game status
    reward = 0
    game_length = 0
    for row in board:
        for c in row:
            if (c != 0):
                game_length += 1

    # defining every action reward
    # if action connect 3 and not getting blocked , reward+=10
    # if action conncet 4 reward += 100
    # if board is at early game , center reward+=4
    # if board is enemy winnable , reward -=550
    if (col == 3 and game_length <= 6):
        reward += 4
    enemy = -player
    # horizontal
    win = 0
    for row in range(rowLen):
        row_now = board[row]
        if (win == 1):
            break
        for i in range(4):
            window = row_now[i:i + 4]
            count_dict = collections.Counter(window)
            if (count_dict[player] == 3 and count_dict[posEmpty] == 1):
                reward += 8
            elif (count_dict[player] == 4):
                reward += 100
                win = 1
                break
            if (count_dict[enemy] == 2 and count_dict[posEmpty] == 2):
                reward -= 2
            elif (count_dict[enemy] == 3 and count_dict[posEmpty] == 1):
                reward -= 4
                for j in range(4):
                    if (window[j] == posEmpty):
                        if row==0:
                            reward -= 150
                            return reward
                        elif (board[row - 1][j] != 0):
                             reward -= 150
                             return reward
    # vertical
    for col in range(colLen):
        col_now = board[:, col]
        if (win == 1):
            break
        for i in range(3):
            window = col_now[i:i + 4]
            count_dict = collections.Counter(window)
            if (count_dict[player] == 3 and count_dict[posEmpty] == 1):
                reward += 8
            elif (count_dict[player] == 4):
                reward += 100
                win = 1
                return reward
            if (count_dict[enemy] == 2 and count_dict[posEmpty] == 2):
                reward -= 2
            elif (count_dict[enemy] == 3 and count_dict[posEmpty] == 1):
                reward -= 150
    # positive diagonal
    for r in range(rowLen - 3):
        if (win == 1):
            break
        for c in range(colLen - 3):
            window = [board[r + i][c + i] for i in range(4)]
            count_dict = collections.Counter(window)
            if (count_dict[player] == 3 and count_dict[posEmpty] == 1):
                reward += 8
            elif (count_dict[player] == 4):
                reward += 100
                win = 1
                return reward

            if (count_dict[enemy] == 2 and count_dict[posEmpty] == 2):
                reward -= 2
            elif (count_dict[enemy] == 3 and count_dict[posEmpty] == 1):
                reward -= 4
                for j in range(4):
                    if (board[r + j][c + j] == 0):
                        if ((r + j) > 0):
                            if (board[r + j - 1][c + j] != 0):
                                reward -= 150
                                return reward
                        elif ((r + j) == 0):
                            reward -= 150
                            return reward

    # negative diagonal
    for r in range(rowLen - 3):
        if (win == 1):
            break
        for c in range(colLen - 3):
            window = [board[r + 3 - i][c + i] for i in range(4)]
            count_dict = collections.Counter(window)
            if (count_dict[player] == 3 and count_dict[posEmpty] == 1):
                reward += 8
            elif (count_dict[player] == 4):
                reward += 100
                win = 1
                return reward

            if (count_dict[enemy] == 2 and count_dict[posEmpty] == 2):
                reward -= 2
            elif (count_dict[enemy] == 3 and count_dict[posEmpty] == 1):
                reward -= 4
                for j in range(4):
                    if (board[r + 3 - j][c + j] == 0):
                        if ((r + 3 - j) > 0):
                            if (board[r + 2 - j][c + j] != 0):
                                reward -= 150
                                return reward

                        elif ((r + 3 - j) == 0):
                            reward -= 150
                            return reward

    # action=select of column ?
    return reward

def get_next_open_row(board, col):
    for r in range(rowLen):
        if board[r][col] == 0:
            return r

def is_terminal_node(board):
    return winning_move(board, 1) or winning_move(board, -1) or len(get_valid_columns(board)) == 0
def minimax(board, depth, alpha, beta, maximizingPlayer,player,col):
    valid_locations = get_valid_columns(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, player):
                return (None, 100000000000000)
            elif winning_move(board, -player):
                return (None, -10000000000000)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            print("reward=",get_reward(board, player,col),"board=",board ,"player=",player)
            return (None, get_reward(board, player,col))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, player)
            new_score = minimax(b_copy, depth-1, alpha, beta, False,-player,col)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else: # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)
            drop_piece(b_copy, row, col, player)
            new_score = minimax(b_copy, depth-1, alpha, beta, True,-player,col) [1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def chessBoardAIUpdate(chessBoard, player):
    board = copy.deepcopy(chessBoard)
    game_length=0
    model_move = True

    for row in board:
        for c in row:
            if (c != 0):
                game_length += 1

    valid_col = get_valid_columns(board)
    # if(random.random()> 0.9):
    #     model = keras.models.load_model('modelp1.h5')
    #     predict = model.predict(np.array(board).reshape(1,42))
    #     board = drop_board(board,predict,player)
    #     if predict not in valid_col:
    #         if game_length > 5:
    #             col, minimax_score = minimax(board, 6, -math.inf, math.inf, True, player)
    #             chessBoard = drop_board(chessBoard, col, player)
    #         else:
    #             col, minimax_score = minimax(board, 4, -math.inf, math.inf, True, player)
    #             chessBoard = drop_board(chessBoard, col, player)
    #     else:
    #         for i in valid_col:
    #             if (winning_move(drop_board(copy.deepcopy(board), i, -player), -player)) :
    #                 model_move=False
    #                 break
    #         if model_move:
    #             chessBoard = drop_board(chessBoard,predict,player)
    #         else:
    #             if game_length > 5:
    #                 col, minimax_score = minimax(board, 6, -math.inf, math.inf, True, player)
    #                 chessBoard = drop_board(chessBoard, col, player)
    #             else:
    #                 col, minimax_score = minimax(board, 4, -math.inf, math.inf, True, player)
    #                 chessBoard = drop_board(chessBoard, col, player)
    #
    # else:
    #
    #     if game_length > 5:
    #         col, minimax_score = minimax(board, 6, -math.inf, math.inf, True, player)
    #         chessBoard = drop_board(chessBoard, col, player)
    #     else:
    #         col, minimax_score = minimax(board, 2, -math.inf, math.inf, True, player)
    #         chessBoard = drop_board(chessBoard, col, player)

    model = keras.models.load_model('modelp1.h5')
    output = model.predict(np.array(board).reshape(1,42))
    predict = np.argmax(output)
    print(predict)
    if game_length<5:
        drop_board(chessBoard,3,player)
    else:
        if predict not in valid_col:
            drop_board(chessBoard,random.choice(valid_col),player)
        else:
            drop_board(chessBoard,predict,player)

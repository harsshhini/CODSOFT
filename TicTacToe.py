import numpy as np

# starting declarations 
WIN = 1000
DRAW = 0
LOSS = -1000
AI_MARKER = 'O'
PLAYER_MARKER = 'X'
EMPTY_SPACE = '-'
START_DEPTH = 0

# possible winning states
winning_states = [
    # column
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    # row
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    # diagonal
    [(0, 0), (1, 1), (2, 2)],
    [(2, 0), (1, 1), (0, 2)]
]

# prints game states - win,draw or loss
def print_game_state(state):
    if state == WIN:
        print("WIN")
    elif state == DRAW:
        print("DRAW")
    elif state == LOSS:
        print("LOSS")

# prints board with the current entries of user and AI
def print_board(board):
    print()
    print(board[0][0], "|", board[0][1], "|", board[0][2])
    print("----------")
    print(board[1][0], "|", board[1][1], "|", board[1][2])
    print("----------")
    print(board[2][0], "|", board[2][1], "|", board[2][2])
    print()

# gets all non occupied places of board
def get_nonOccupied_positions(board):
    legal_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] != AI_MARKER and board[i][j] != PLAYER_MARKER:
                legal_moves.append((i, j))
    return legal_moves

# checks if a position is occupied
def position_occupied(board, pos):
    legal_moves = get_nonOccupied_positions(board)
    for move in legal_moves:
        if pos[0] == move[0] and pos[1] == move[1]:
            return False
    return True

# checks of the positions occupied by a given marker
def get_occupied_positions(board, marker):
    occupied_positions = []
    for i in range(3):
        for j in range(3):
            if marker == board[i][j]:
                occupied_positions.append((i, j))
    return occupied_positions

# checks if the board is full
def board_is_full(board):
    legal_moves = get_nonOccupied_positions(board)
    if len(legal_moves) == 0:
        return True
    else:
        return False

# checks if the game has been won
def game_is_won(occupied_positions):
    game_won = False
    for i in range(len(winning_states)):
        game_won = True
        curr_win_state = winning_states[i]
        for j in range(3):
            if curr_win_state[j] not in occupied_positions:
                game_won = False
                break
        if game_won:
            break
    return game_won

# returns whos next opponent(AI or PLAYER)
def get_opponent_marker(marker):
    opponent_marker =""
    if marker == PLAYER_MARKER:
        opponent_marker = AI_MARKER
    else:
        opponent_marker = PLAYER_MARKER
    return opponent_marker
 
# Checks if someone has lost,drawn or won
def get_board_state(board, marker):
    opponent_marker = get_opponent_marker(marker)
    occupied_positions = get_occupied_positions(board, marker)
    is_won = game_is_won(occupied_positions)
    if is_won:
        return WIN
    occupied_positions = get_occupied_positions(board, opponent_marker)
    is_lost = game_is_won(occupied_positions)
    if is_lost:
        return LOSS
    is_full = board_is_full(board)
    if is_full:
        return DRAW
    return DRAW

'''                                   *** MAIN PART ***
    Minimax algorithm with alpha-beta pruning for optimizing the decision-making process in a two-player game
    with a maximizing player (AI) and a minimizing player (opponent).'''
def minimax_optimization(board, marker, depth, alpha, beta):
    ''' depth: The depth of the current search in the game tree.
        alpha: The best score that the maximizing player (AI) can guarantee.
        beta: The best score that the minimizing player (opponent) can guarantee.'''
    
    best_move = (-1, -1)    # intializing best move
    best_score = LOSS if marker == AI_MARKER else WIN   # best score based on whether it's the AI's turn or the opponent's turn.
    
    # If we hit a terminal state (leaf node), return the best score and move
    if board_is_full(board) or DRAW != get_board_state(board, AI_MARKER):
        best_score = get_board_state(board, AI_MARKER)
        return (best_score, best_move)
    legal_moves = get_nonOccupied_positions(board)
    for curr_move in legal_moves:
        board[curr_move[0]][curr_move[1]] = marker
        
        # Maximizing player's turn
        if marker == AI_MARKER:
            score = minimax_optimization(board, PLAYER_MARKER, depth + 1, alpha, beta)[0]
            '''Recursively calls the minimax_optimization function for the opponent's turn (minimizing player)
              and retrieves the score of the best move.'''
            
            if best_score < score:  # Get the best scoring move
                best_score = score - depth * 10    
                '''Updates the best score by subtracting a penalty based on the depth of the current move.
                  This is done to prefer shorter paths to victory.''' 
                
                best_move = curr_move
                
                # Check if this branch's best move is worse than the best
                # option of a previously search branch. If it is, skip it
                alpha = max(alpha, best_score)
                board[curr_move[0]][curr_move[1]] = EMPTY_SPACE     #Undoes the move for backtracking.
                if beta <= alpha:   # Prunes the remaining branches of the search tree if beta is less than or equal to alpha.
                    break
        
        # Minimizing opponent's turn
        else:   
            score = minimax_optimization(board, AI_MARKER, depth + 1, alpha, beta)[0]
            if best_score > score:
                best_score = score + depth * 10
                best_move = curr_move
                
                # check if this branch's best move is worse than the best
				# option of a previously search branch. If it is, skip it
                beta = min(beta, best_score)    # Ensures that the board is restored to its original state after exploring a move.
                board[curr_move[0]][curr_move[1]] = EMPTY_SPACE
                if beta <= alpha:
                    break
        board[curr_move[0]][curr_move[1]] = EMPTY_SPACE     # UNDO move
    return (best_score, best_move) # Returns the best score and move found during the search.

# checks if the game is finished
def game_is_done(board):
    if board_is_full(board):
        return True

    if DRAW != get_board_state(board, AI_MARKER):
        return True

    return False



board = [[EMPTY_SPACE] * 3 for _ in range(3)]

print("********************************\n\n\tTic Tac Toe AI\n\n********************************\n\n")
print("Player = X\t AI Computer = O\n\n")

print_board(board)

while not game_is_done(board):
    row = int(input("Row play: "))
    col = int(input("Col play: "))

    if position_occupied(board, (row, col)):
        print(f"The position ({row}, {col}) is occupied. Try another one...")
        continue
    else:
        board[row][col] = PLAYER_MARKER

    ai_move = minimax_optimization(board, AI_MARKER, START_DEPTH, LOSS, WIN)

    board[ai_move[1][0]][ai_move[1][1]] = AI_MARKER

    print_board(board)

print("********** GAME OVER **********\n\n")

player_state = get_board_state(board, PLAYER_MARKER)

print("PLAYER ", end="")
print_game_state(player_state)

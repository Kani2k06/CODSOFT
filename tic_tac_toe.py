 import math

def print_board(board):
    print("\n")
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board):
    
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != ' ':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != ' ':
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]
    return None

def is_full(board):
    return all(cell != ' ' for row in board for cell in row)

def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'O':
        return 1
    elif winner == 'X':
        return -1
    elif is_full(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    best_score = min(score, best_score)
        return best_score

def ai_move(board):
    best_score = -math.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                score = minimax(board, 0, False)
                board[i][j] = ' '
                if score > best_score:
                    best_score = score
                    move = (i, j)
    if move:
        board[move[0]][move[1]] = 'O'

def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print("Tic-Tac-Toe: You (X) vs AI (O)")
    print_board(board)

    while True:
        
        while True:
            try:
                move = input("Enter your move (row col): ").split()
                row, col = int(move[0]), int(move[1])
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    break
                else:
                    print("Cell already taken. Try again.")
            except:
                print("Invalid input. Enter row and column like 0 2")

        print_board(board)

        if check_winner(board) == 'X':
            print("You win!")
            break
        if is_full(board):
            print("It's a draw!")
            break

        
        print("AI's turn...")
        ai_move(board)
        print_board(board)

        if check_winner(board) == 'O':
            print("AI wins!")
            break
        if is_full(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    play_game()

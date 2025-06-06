from parser.make_state import load_batch
# If running independantly
# from make_state import load_batch

def get_state(layers):
    state = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(len(state)):
        for j in range(len(state)):
            state[i][j] = layers[2][i][j]-layers[0][i][j]
    return state

def positions_layer(state, type=0):
    layer = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(len(state)):
        for j in range(len(state)):
            layer[i][j] = 1 if state[i][j]==type else 0
    return layer

def print_board(layer):
    for j in range(8):
        print(layer[j])

def print_move(move):
    row, col = move
    print(f'({row+1}, {col+1})')

if __name__ == "__main__":
    gen = load_batch('./parser/all_games.txt', batch_size=1)
    input, moves = next(gen)

    # layers = []
    # for i in range(3):
    #     layers.append(positions_layer(input[0], i-1))
    #     print_board(layers[i])
    #     i+=1
    # layer = get_state(layers)
    # print_board(layer)

    print_board(input[0])
    print_move(moves[0])

    
    

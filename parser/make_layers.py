from parser.make_state import load_batch
# If running independantly
# from make_state import load_batch

def positions_layer(state, type=0):
    layer = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(len(state)):
        for j in range(len(state)):
            layer[i][j] = 1 if state[i][j]==type else 0
    return layer


if __name__ == "__main__":
    states, moves = load_batch('./parser/all_games.txt', batch_size=1, batch_idx=0)

    for i in range(20):
        layer = positions_layer(states[0][i], 0)
        for j in range(8):
            print(layer[j])
        print('\n')
    
    

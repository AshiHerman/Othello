import re

cols = "ABCDEFGH"
rows = "01234567"

def parse_move(move):
    if len(move) != 2:
        return None  # Invalid move length
    if move[0] not in cols or move[1] not in rows:
        return None  # Invalid move chars
    col = cols.index(move[0])
    row = int(move[1])
    return (col, row)

def move_str(col, row):
    if 0 <= col < 8 and 0 <= row < 8:
        return cols[col] + str(row)
    else:
        return None

def rotate_90(col, row):
    return (7 - row, col)
def rotate_180(col, row):
    return (7 - col, 7 - row)
def rotate_270(col, row):
    return (row, 7 - col)

def augment_game(moves):
    parsed = []
    for i in range(0, len(moves), 2):
        move = parse_move(moves[i:i+2])
        if move is not None:
            parsed.append(move)
    games = []
    for rot in [lambda c, r: (c, r), rotate_90, rotate_180, rotate_270]:
        rotated = [rot(c, r) for c, r in parsed]
        out_moves = []
        for c, r in rotated:
            ms = move_str(c, r)
            if ms is not None:
                out_moves.append(ms)
        games.append(''.join(out_moves))
    return games

# input_file = "./Othello-Board-Parser/all_games.txt"
# output_file = "./Othello-Board-Parser/augmented_games.txt.txt"

# with open(input_file, "r", encoding="utf-8") as inp, open(output_file, "w", encoding="utf-8") as out:
#     for line in inp:
#         line = line.strip()
#         if not line or len(line) % 2 != 0:
#             continue  # skip empty/bad lines
#         for aug_game in augment_game(line):
#             out.write(aug_game + "\n")

# print(f"Augmented games written to {output_file}")


def test_augment_game():
    print("Running tests for augment_game...")

    # Test: single move in each corner
    tests = {
    "A0": ["A0", "H0", "H7", "A7"],
    "H0": ["H0", "H7", "A7", "A0"],
    "A7": ["A7", "A0", "H0", "H7"],
    "H7": ["H7", "A7", "A0", "H0"],
    }

    for orig, expected in tests.items():
        results = augment_game(orig)
        assert results == expected, f"Failed: {orig} → {results} (expected {expected})"
        print(f"Pass: {orig} rotations → {results}")

    # Test: a mini-game string
    # E.g., moves = "A0B1C2" (should rotate each position accordingly)
    orig = "A0B1C2"
    results = augment_game(orig)
    expected = [
        "A0B1C2",        # Original
        "A7B0C1",        # 90°
        "H7G6F5",        # 180°
        "H0G7F6"         # 270°
    ]
    print(f"Test: {orig} → {results}")

    # Test: invalid move string
    invalid = "ZZ"  # not a valid board move
    results = augment_game(invalid)
    assert all(r == "" for r in results), "Invalid moves should be filtered out"
    print("Pass: invalid moves handled")

    # Test: incomplete move
    incomplete = "A"
    results = augment_game(incomplete)
    assert all(r == "" for r in results), "Incomplete moves should be filtered out"
    print("Pass: incomplete moves handled")

    print("All tests passed!")

if __name__ == "__main__":
    # Run tests
    test_augment_game()

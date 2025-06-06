import random

# Read all lines from the file
with open('./parser/all_ordered_games.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Shuffle the lines in place
random.shuffle(lines)

# Write the shuffled lines to a new file (or overwrite the original)
with open('./parser/all_games.txt', 'w', encoding='utf-8') as f:
    f.writelines(lines)

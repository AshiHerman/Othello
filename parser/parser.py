import os
import glob
import struct

# Set the folder containing .wtb files
folder = "wthor-database"
CLEAN = True

if __name__ == "__main__":
    # Find all .wtb files (non-recursive)
    wtb_files = glob.glob(os.path.join(folder, "*.wtb")) + glob.glob(os.path.join(folder, "*.WTB"))

    # If you want to search subfolders too, use:
    # wtb_files = glob.glob(os.path.join(folder, "**", "*.wtb"), recursive=True)
    # wtb_files += glob.glob(os.path.join(folder, "**", "*.WTB"), recursive=True)

    for fn in wtb_files:
        outputfile = fn[:-4] + ".txt" if fn.lower().endswith(".wtb") else fn + ".txt"
        if os.path.exists(outputfile):
            continue
        print(f"Processing {fn}")
        with open(fn, "rb") as f:
            with open(outputfile, "w", encoding="utf-8") as out:
                header = f.read(16)
                if len(header) < 8:
                    continue
                # Unpack number of games from bytes 4-7
                hands = struct.unpack("<I", header[4:8])[0]

                for _ in range(hands):
                    game = f.read(68)
                    if len(game) < 68:
                        break
                    try:
                        moves = list(game[8:68])
                        line = ""
                        for h in moves:
                            col = "_ABCDEFGH"[h % 10]
                            row = str(h // 10)
                            move = col + row
                            line += move
                        line = line.rstrip("_0")
                        if not CLEAN or "_0" not in line:
                            out.write(line + "\n")
                    except Exception as e:
                        print("Error in file, ignoring:", moves)
                        break

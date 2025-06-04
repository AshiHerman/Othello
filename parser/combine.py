import os
import glob

if __name__ == "__main__":
    folder = "wthor-database"
    output_file = "all_games.txt"

    # Find all .txt files, skipping the combined output file if it's in the same folder
    txt_files = glob.glob(os.path.join(folder, "*.txt"))

    with open(output_file, "w", encoding="utf-8") as out:
        for fn in txt_files:
            print(f"Adding games from {fn}")
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    # Remove any empty lines just in case
                    line = line.strip()
                    if line:
                        out.write(line + "\n")

    print(f"Combined all games into {output_file}")

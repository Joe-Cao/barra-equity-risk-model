from pathlib import Path
import shutil

# ========= 1. Modify here: set your paths =========
# Assume the following three items are under the same folder on your Desktop:
#   nasdaq stocks/
#   nyse stocks/
#   barra_sp500_500_expected_filenames.txt

base_dir = Path(r"C:\Users\caoxi\python练习\Barra")  # Change this to your actual path

nasdaq_dir = base_dir / "nasdaq stocks"
nyse_dir = base_dir / "nyse stocks"

expected_file = base_dir / "barra_sp500_500_expected_filenames.txt"
target_dir = base_dir / "target stocks"

# ========= 2. Create the target stocks folder =========
target_dir.mkdir(exist_ok=True)

# ========= 3. Read the target stock filenames =========
with open(expected_file, "r", encoding="utf-8") as f:
    target_filenames = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Number of target filenames: {len(target_filenames)}")

# ========= 4. Build the source file index =========
# key: lowercase filename
# value: full path

source_dirs = [nasdaq_dir, nyse_dir]
file_index = {}

for folder in source_dirs:
    if not folder.exists():
        print(f"Warning: folder does not exist: {folder}")
        continue

    for file_path in folder.glob("*.txt"):
        file_index[file_path.name.lower()] = file_path

print(f"Number of available source files: {len(file_index)}")

# ========= 5. Copy the target files =========
found = []
missing = []

for filename in target_filenames:
    key = filename.lower()

    if key in file_index:
        src = file_index[key]
        dst = target_dir / src.name

        shutil.copy2(src, dst)
        found.append(filename)
    else:
        missing.append(filename)

# ========= 6. Save the missing file list =========
missing_file_path = target_dir / "missing_files.txt"

with open(missing_file_path, "w", encoding="utf-8") as f:
    for filename in missing:
        f.write(filename + "\n")

# ========= 7. Print the results =========
print("Done.")
print(f"Copied files: {len(found)}")
print(f"Missing files: {len(missing)}")
print(f"Target folder: {target_dir}")
print(f"Missing list saved to: {missing_file_path}")
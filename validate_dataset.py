import os

# ================= CONFIGURATION =================
# Update these paths to match your actual files
LABEL_FILE = "../train_data/rec/train.txt" # "./train_data/rec/train.txt"
DATA_ROOT = "../train_data/rec/train/"  # Root folder where images are stored
DICT_FILE = "../PaddleOCR/ppocr/utils/dict/ppocrv5_arabic_dict.txt" #"./ppocr/utils/dict/ppocrv5_arabic_dict.txt"
# MUST match the 'max_text_length' in your YAML file (Global section)
MAX_TEXT_LENGTH = 25
# =================================================

def load_dictionary(dict_path):
    if not os.path.exists(dict_path):
        print(f"❌ Error: Dictionary file not found at {dict_path}")
        return None
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = f.read().splitlines()
    
    char_set = set(chars)
    char_set.add(" ") 
    print(f"✅ Loaded Dictionary: {len(char_set)} characters found.")
    return char_set

def validate_dataset(label_file, root_dir, valid_chars, max_len):
    if not os.path.exists(label_file):
        print(f"❌ Error: Label file not found at {label_file}")
        return

    missing_files = []
    missing_chars = set()
    bad_format_lines = []
    long_lines = []  # <--- NEW: Stores lines that are too long
    total_lines = 0

    print(f"\n🔍 Scanning {label_file}...")

    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 1. Check Format
            if "\t" not in line:
                bad_format_lines.append(f"Line {idx+1}")
                continue
            
            path, label = line.split("\t", 1)

            # 2. Check File Existence
            full_path = os.path.join(root_dir, path)
            if not os.path.exists(full_path):
                missing_files.append(path)

            # 3. Check Characters
            for char in label:
                if char not in valid_chars:
                    missing_chars.add(char)

            # 4. Check Length (NEW)
            if len(label) > max_len:
                long_lines.append((idx+1, len(label), label[:30] + "..."))

    # ================= REPORT =================
    print("\n" + "="*40)
    print(f"📊 VALIDATION REPORT FOR: {label_file}")
    print("="*40)
    print(f"✅ {total_lines} Labels found")
    # 1. Format
    if bad_format_lines:
        print(f"❌ FORMAT ERRORS: {len(bad_format_lines)} lines missing tabs.")
    else:
        print(f"✅ Formatting: OK")

    # 2. Files
    if missing_files:
        print(f"❌ MISSING IMAGES: {len(missing_files)} files not found.")
    else:
        print(f"✅ Files: OK")

    # 3. Length (Crucial for your issue)
    if long_lines:
        print(f"⚠️  LENGTH WARNING: {len(long_lines)} labels exceed max_text_length ({max_len}).")
        print(f"   These labels will be TRUNCATED during training (bad for accuracy).")
        print(f"   Examples (Line #, Length):")
        for i in range(min(100, len(long_lines))):
            print(f"     - Line {long_lines[i][0]}: {long_lines[i][1]} chars -> '{long_lines[i][2]}'")
        print(f"👉 ACTION: Either increase 'max_text_length' in YAML or shorten these labels.")
    else:
        print(f"✅ Length: All labels are under {max_len} characters.")

    # 4. Dictionary
    if missing_chars:
        print(f"⚠️  MISSING CHARACTERS: {len(missing_chars)} chars not in dictionary.")
        print(f"   Missing Set: {sorted(list(missing_chars))}")
    else:
        print(f"✅ Dictionary: OK")

    print("="*40 + "\n")

if __name__ == "__main__":
    valid_chars = load_dictionary(DICT_FILE)
    if valid_chars:
        validate_dataset(LABEL_FILE, DATA_ROOT, valid_chars, MAX_TEXT_LENGTH)
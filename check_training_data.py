# Check the first line of your train.txt
filename = "/home/sara/data/capmas/vsworkspace/capmas_projects/train_data/rec/train.txt"

with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()
    
for line in lines:
    # Split the image path from the label (assuming tab or space separator)
    # Adjust split logic based on your file format
    parts = line.strip().split('\t') 
    
    if len(parts) >= 2:
        image_path = parts[0]
        label_text = parts[1]
        
        print(f"Image: {image_path}")
        print(f"Label (Display): {label_text}")
        print(f"Label (Raw List): {list(label_text)}")
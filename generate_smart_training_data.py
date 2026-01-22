# pip install arabic-reshaper python-bidi
# sudo apt update
# sudo apt install fonts-amiri
# sudo apt install ttf-mscorefonts-installer

import os
import shutil
import random
from PIL import Image, ImageDraw, ImageFont
# import arabic_reshaper
# from bidi.algorithm import get_display
from  libs.mytools import convert_to_eastern_arabic
import random
import numpy as np
import logging
import validate_dataset
import textwrap


# Configure logging (usually done at the start of your script)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- CONFIG ---
CAPMAS_TRAINING_GT_DIR= '/home/sara/data/capmas/vsworkspace/capmas_training_GT/ocr_training_data_5K_0padd_50size'
PARENT_DIR = '/home/sara/data/capmas/vsworkspace/capmas_projects/PPOCRLabel/train_data' #"train_data/tatweel_data"
IMG_DIR_NAME = 'crop_img'
FONT_PATH = [
            "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
            "/usr/share/fonts/Amiri/Amiri-Regular.ttf",
            # "/usr/share/fonts/traditional-arabic-morph-regular/traditional-arabic-morph-regular.ttf",
            # "/home/sara/data/capmas/vsworkspace/capmas_projects/PPOCRLabel/fonts/traditional-arabic.ttf",
            "/home/sara/data/capmas/vsworkspace/capmas_projects/PPOCRLabel/fonts/Bahij_Lotus/Bahij_Lotus-Light.ttf",
            "/home/sara/data/capmas/vsworkspace/capmas_projects/PPOCRLabel/fonts/Bahij_Yakout/Bahij_Yakout-Regular.ttf",  #Perfect for 1991
            #"/home/sara/data/capmas/vsworkspace/capmas_projects/PPOCRLabel/fonts/29LT-Azer-Bold.otf",  #No
            #"/home/sara/data/capmas/vsworkspace/capmas_projects/PPOCRLabel/fonts/Alyamamawght.ttf",  #No
            # "/home/sara/Downloads/Parastoo-Bold.ttf", #nice but not perfect specially lam Alef
            ]
IMAGE_HEIGHT = 48
NUM_IMAGES = 100
AUGMENTATION_FACTOR=50

#Define letters that CAN accept a Tatweel after them
# We exclude: ا د ذ ر ز و ؤ ء ة and ends of words
#We also exclude the letter lam as it cause lots of troubles with lam alef'ل'
VALID_PREDECESSORS = set("بتثجحخسشصضطظعغفقكمنهيئ")

SPACER = " "
MAX_LINE_LENGTH=25 #100  #max number of character per line after adding tatweels
MAX_TATWEEL_STRETCH=0 #6 # Maximum random number of tatweel character added per word





def generate_image_and_label(text_raw, filename, font):
    #Convert English numbers to Arabi (Hindi)
    text_raw = convert_to_eastern_arabic(text_raw)

    # # 1. Reshape Arabic (Fixes ligatures and letter forms)
    # reshaped_text = arabic_reshaper.reshape(text_raw)
    # # # 2. Reorder for RTL (Right-to-Left) display
    # bidi_text = get_display(reshaped_text)

    # bidi_text = 'هذا يعكس الاهمية الحيوية لشبه جزيرة سيناء'#text_raw
    bidi_text = text_raw

    # Calculate text size
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_w = bbox[2] - bbox[0] # <--- Calculate width
    text_h = bbox[3] - bbox[1] # <--- Calculate height

    
    # Create image with random extra padding for "Space" training
    width = text_w + random.randint(4, 30) #random.randint(50, 150)
    image = Image.new('RGB', (width, IMAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Calculate positions
    # Center text vertically, random horizontal position
    x_pos = (width - text_w) // 2

    # Calculate the vertical center
    # Formula: (Container Height - Text Height) / 2
    y_pos = (IMAGE_HEIGHT - text_h) // 2

    # Draw the text with offset correction
    # We subtract bbox[1] because the bounding box top might not be 0 relative to the baseline
    draw.text((x_pos, y_pos - bbox[1]), bidi_text, font=font, fill=(0, 0, 0))

    image.save(f"{PARENT_DIR}/{IMG_DIR_NAME}/{filename}")
    
    # Return label
    return f"{IMG_DIR_NAME}/{filename}\t{text_raw}"



def generate_year_ranges_images(font, num_images, start_year, end_year):
    #Generates year ranges images
    labels = []
    for i in range(num_images):
        year_from = random.randint(start_year, end_year-10)
        year_to = random.randint(year_from+1, end_year)
        # Randomly picks either a space or nothing
        separator = random.choice([SPACER, ''])
        #Concatenate the text of year ranges
        full_text = f"{year_from}{separator}-{separator}{year_to}"        
        #write results into file
        filename = f"year_range_{font.getname()[0]}_{i}.png"
        line = generate_image_and_label(full_text, filename, font)
        if line: 
            labels.append(line)
    return labels



def generate_decimals_arabic_images(font, num_images, start_num, end_num, max_precision=4):
    #Generates year ranges images
    labels = []
    numbers = np.random.uniform(start_num, end_num, num_images)   
    for i, num in enumerate(numbers[0:num_images]):   
        precision = random.randint(0, max_precision)        
        text = str(round(num, precision))
        separator = random.choice([',', 'ر'])
        text = text.replace('.', separator)
        filename = f"decimals_{start_num}_{end_num}_{font.getname()[0]}_{i}.png"
        line = generate_image_and_label(text, filename, font)
        if line: 
            labels.append(line)
    return labels
        

def read_ground_truth(folder_path):
    text_contents = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_contents.append(file.read())
    return text_contents



# def insert_one_tatweel_per_word(text, base_probability=0.05, max_stretch=MAX_TATWEEL_STRETCH, max_line_length=MAX_LINE_LENGTH):
#     """
#     Inserts AT MOST ONE tatweel per word, heavily favoring the last quarter.
#     """   
    
#     if len(text.strip())==0:
#         return None
#     # if text is only 2 characters, do not stretch, or change return it as is
#     elif len(text.strip()) < 3:
#         return text
    
#     tokens = re.split(r'(\s+)', text)
#     processed_tokens = []  
    
#     for token in tokens:
#         # Skip empty or whitespace tokens
#         if not token.strip():
#             processed_tokens.append(token)
#             continue
        
#         new_word = []
#         word_len = len(token)
        
#         # Track if we have already stretched this specific word
#         word_has_tatweel = True  if "ـ" in token else False
        
        
#         for i, char in enumerate(token):
#             new_word.append(char)
            
#             # 1. Check if char allows stretching
#             # 2. Check if we haven't already stretched this word
#             # 3. Check bounds (not the last letter)
#             # 4. Check if in the last half of the word
#             if max_stretch>0 and (char in VALID_PREDECESSORS) and (not word_has_tatweel) and (i + 1 < word_len)  and (i + 1 > word_len/2):
                
#                 next_char = token[i+1]
                
#                 # Ensure next char is valid (no stretching before punctuation)
#                 if next_char.isalnum(): 
                    
#                     # --- CUBIC WEIGHTING LOGIC ---
#                     progress = i / max(1, word_len - 1)
#                     cubic_progress = progress ** 3 
                    
#                     # Calculate dynamic probability
#                     dynamic_prob = base_probability + ((1.0 - base_probability) * cubic_progress)
                    
#                     # Roll the dice
#                     if random.random() < dynamic_prob:
#                         stretch_length = random.randint(1, max_stretch)
#                         new_word.append("ـ" * stretch_length)
                        
#                         # IMPORTANT: Lock this word so no more tatweels are added
#                         word_has_tatweel = True
        
#         # adds up the lengths of each individual string including the current one
#         total_chars = sum(len(s) for s in processed_tokens) +  sum(len(s) for s in new_word)
#         #If still within range, add current word to the list       
#         if total_chars < max_line_length:
#             processed_tokens.append("".join(new_word))
#         #otherwise, just send this part of the line
#         else:
#             return "".join(processed_tokens)
    
#     #If list of words finished, just send it
#     return "".join(processed_tokens)



def split_labels(input_lines, max_length):
    """
    Splits a list of text lines into a longer list where each line 
    is no longer than 'max_length'.
    
    Args:
        input_lines (list): List of original strings.
        max_length (int): Maximum allowed characters per line.
        
    Returns:
        list: A flattened list of split strings.
    """
    output_lines = []
    
    for line in input_lines:
        
        # textwrap.wrap returns a list of strings for the single input line
        # We extend our main output list with these new smaller lines
        wrapped_lines = textwrap.wrap(line, width=max_length, break_long_words=True)
        
        # If the line was empty, textwrap might return empty list; 
        # optional: preserve empty lines if needed
        if not wrapped_lines and line == "":
            output_lines.append("")
        else:
            output_lines.extend(wrapped_lines)
            
    return output_lines



def generate_ground_truth_text(text_contents, number_images):
    # Create a new list with augmented text
    label_contents = []

    for i,line in enumerate(text_contents[0:number_images]):   
        if len(line.strip())>0:
            #write results into file
            filename = f"gt_text_{font.getname()[0]}_{i}.png"

            label_line = generate_image_and_label(line, filename, font)
            if label_line: 
                label_contents.append(label_line)

    return label_contents



if __name__ == "__main__":

    # Check if it exists to avoid FileNotFoundError
    if os.path.exists(f"{PARENT_DIR}"):
        shutil.rmtree(f"{PARENT_DIR}")
        print(f"Deleted {f"{PARENT_DIR}"} and all its contents.")
    
    os.makedirs(f'{PARENT_DIR}/{IMG_DIR_NAME}')
    
    labels = []

    #Ensure fonts exists
    for font_path in FONT_PATH:
        try:
            font = ImageFont.truetype(font_path, 32)
            logging.info(f"Generating Smart Trainning Data: (MAX_LINE_LENGTH={MAX_LINE_LENGTH}, MAX_TATWEEL_STRETCH={MAX_TATWEEL_STRETCH}, IMAGE_HEIGHT={IMAGE_HEIGHT})")
            
            logging.info(f"Generating {NUM_IMAGES} year ranges images (font: {font.getname()[0]})...")
            years_labels = generate_year_ranges_images(font, NUM_IMAGES, 1900, 2000)
            logging.info(f' >> {len(years_labels)} lines generated successfully ...')
            labels.extend(years_labels)

            logging.info(f"Generating big decimal arabic numbers (font: {font.getname()[0]})...")
            decimal_arabic_labels = generate_decimals_arabic_images(font, NUM_IMAGES, 1000, 10000,2)
            logging.info(f' >> {len(decimal_arabic_labels)} lines generated successfully ...')
            labels.extend(decimal_arabic_labels)

            logging.info(f"Generating small decimal arabic numbers (font: {font.getname()[0]})...")
            decimal_arabic_labels = generate_decimals_arabic_images(font, NUM_IMAGES, 100, 1000, 2)
            logging.info(f' >> {len(decimal_arabic_labels)} lines generated successfully ...')
            labels.extend(decimal_arabic_labels)

            logging.info(f"Generating tiny decimal arabic numbers (font: {font.getname()[0]})...")
            decimal_arabic_labels = generate_decimals_arabic_images(font, NUM_IMAGES, 0, 100, 1)
            logging.info(f' >> {len(decimal_arabic_labels)} lines generated successfully ...')
            labels.extend(decimal_arabic_labels)


            logging.info(f"Generating ground truth labels images (font: {font.getname()[0]})...")
            gt_labels = read_ground_truth(CAPMAS_TRAINING_GT_DIR)
            splitted_gt_labels = split_labels(gt_labels, MAX_LINE_LENGTH)
            text_gtlabels = generate_ground_truth_text(splitted_gt_labels, NUM_IMAGES*AUGMENTATION_FACTOR)
            logging.info(f' >> {len(text_gtlabels)} lines generated successfully ...')
            labels.extend(text_gtlabels)

        except BaseException as e:
            # This automatically logs the error AND the line number/traceback
            logging.exception("An error occurred during processing")    
 
    # Save Label File
    with open(f"{PARENT_DIR}/Label.txt", "w", encoding="utf-8") as f:
        for l in labels:
            f.write(l + "\n")

    logging.info(f"Done! Check '{PARENT_DIR}/Label.txt'")


    #VALIDATING TRAINING Data
    logging.info(f"########################################################")
    logging.info(f"Start validating training data ...")
    valid_chars = validate_dataset.load_dictionary(validate_dataset.DICT_FILE)
    if valid_chars:
        validate_dataset.validate_dataset(f"{PARENT_DIR}/Label.txt", f"{PARENT_DIR}/", valid_chars, MAX_LINE_LENGTH)

    logging.info(f"Validation Done.")
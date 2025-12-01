import cv2
import numpy as np
import logging
import re 
logger = logging.getLogger("PPOCRLabel")




#SKS Created to reverse arabic 
def generate_rtl_label(text: str) -> str:
    """
    Creates the correct visual-order label for Bidi text (Arabic/RTL + Numbers/English/LTR).

    This matches the LTR scan of the OCR model by:
    1. Reversing the entire string.
    2. Finding all LTR sequences (English letters, numbers, and punctuation [.,-]).
    3. Reversing just those LTR sequences back to their LTR order.
    """
    
    # SKS Gemini: Handle newline characters to keep them at the end.
    has_newline = text.endswith('\n')
    if has_newline:
        # Process the text without the newline
        text = text.rstrip('\n')

    # 1. Reverse the whole string
    # "Part-123 is 50 درهم" -> "مهرد 05 si 321-traP"
    reversed_text = text[::-1]

    
    # 2. Find all LTR (English/Number) blocks.
    # This regex matches any sequence of letters, numbers, and [.,-]
    ltr_pattern = re.compile(r'[a-zA-Z0-9\.,-]+')
    
    # 3. This function takes a regex match (e.g., "321-traP")
    #    and returns its reverse (e.g., "Part-123")
    def flip_ltr_block_back(match):
        return match.group(0)[::-1]

    # 4. Apply this function to all found LTR blocks
    # "مهرد 05 si 321-traP" -> "مهرد 50 is Part-123"
    correct_label = ltr_pattern.sub(flip_ltr_block_back, reversed_text)
    
    # SKS Gemini: Add the newline back if it was originally there.
    if has_newline:
        return correct_label + '\n'
    else:
        return correct_label

#SKS Added
def convert_to_eastern_arabic(text: str) -> str:
    """
    Converts English digits (0-9) in a string to Eastern Arabic digits (٠-٩).
    """
    english_digits = "0123456789"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    
    # Create the translation table
    translation_table = str.maketrans(english_digits, arabic_digits)
    
    # Apply the translation
    return text.translate(translation_table)



from PIL import Image as PILImage  #Added by Sara
# Corrected my_read_image to load a 3-channel image robustly using PIL, then convert to OpenCV format
def my_read_image(path):
    
    scale = 0.6 #0.6
    try:
        pil_image = PILImage.open(path).convert("RGB") # Ensure 3 channels
        img_np = np.array(pil_image).astype('uint8')
        cv_imageObj = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # PIL is RGB, OpenCV is BGR
    except Exception as e:
        logger.error(f"Failed to load image {path} with PIL in my_read_image: {e}")
        return None # Return None if image loading fails
    if scale < 1:
        cv_imageObj = cv2.resize(cv_imageObj, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv_imageObj

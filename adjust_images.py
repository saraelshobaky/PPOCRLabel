# This script, takes each tiff image from the input folders
# rotate it if it was described in the file osd_90 or osd_270
# then save it as png file in a png folder inside the input folder


from libs.mytools import my_read_image, generate_rtl_label, cv_rotate2
import glob
import os
import cv2


def main():
    folder_path = "training_data2/"
    # Grab all file paths matching the extension
    image_paths = glob.glob(f'{folder_path}/*.tif')

    for path in image_paths:       
        # Read the image (cv2 loads as BGR by default)
        # img = my_read_image(path, scale=1.0)
        img = my_read_image(path, scale=0.6)

        if 'osd90' in path:
            img = cv_rotate2(img, 90)
        elif 'osd270' in path:
            img = cv_rotate2(img, 270)

        #Save new image to a new folder          
        cv2.imwrite(f'{folder_path}/pngs/{os.path.basename(path).replace('osd','R').replace('.tif','.png')}', img)


main()        
        
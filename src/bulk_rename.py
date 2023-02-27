import numpy as np
import cv2

import math
import sys
import json
import csv
import time
import glob

import os
 
def main():
   
    folder = "img/sens_aor_17"
    for count, filename in enumerate(os.listdir(folder)):
        img = f"Results/Experiment_saor.png"
        src =f"{folder}/{filename}/{img}"  # foldername/filename, if .py file is outside folder
        dst =f'{filename}.png'
        
        print(f'{dst=}')
        print(f'{src=}')
        # rename() function will
        # rename all the files
        os.rename(src, dst)
        # print(count)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()
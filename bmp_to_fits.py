#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import os
import glob
from pylab import *
from math import e
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import rotate
from PIL import Image
import pandas as pd
import pathlib
import genieshots_converter

from datetime import datetime


# In[64]:


notes = ("laser and dot weekday test, accelerometers on mirror again, pumps on -> off midday 11-19")


# In[65]:


#path1  = "/Users/anh5866/Desktop/Coding/OGRE/collimated_beam/temp/*.bmp"
#path1  = "/Users/anh5866/Desktop/Coding/OGRE/collimated_beam/20250917/*.bmp"
#path1 = "/Users/anh5866/OneDrive - The Pennsylvania State University/20250917/*.bmp"


#change for each file
foldername_date = "20251118"
foldername = "lasermirror"
rootpath = pathlib.Path(r"C:\Users\jad507\OneDrive - The Pennsylvania State University\Pictures\Reverse Telescope Test")
#change to your path

filepath = rootpath / foldername_date / foldername
path1 = filepath.__str__() + "/*.bmp"

bmp_list = glob.glob(path1)
bmp_list = np.sort(bmp_list,kind='standardsort')

print(bmp_list)
print(len(bmp_list))


# In[66]:


nested_folder_path = rootpath / (foldername_date + "_data")
try:
    os.makedirs(nested_folder_path, exist_ok=True) # exist_ok=True prevents error if already exists
    print(f"Nested folder '{nested_folder_path}' created successfully.")
except Exception as e:
    print(f"An error occurred: {e}")


# In[67]:


nested_nested_folder_path = nested_folder_path / foldername
try:
    os.makedirs(nested_nested_folder_path, exist_ok=True) # exist_ok=True prevents error if already exists
    print(f"Nested folder '{nested_nested_folder_path}' created successfully.")
except Exception as e:
    print(f"An error occurred: {e}")


# In[68]:


filename = os.path.basename(filepath)
num_data = len(bmp_list)


#for i in range(len(bmp_list)):
#    print(str(bmp_list[i][-21:-4]))
    
creation_date_start = str(bmp_list[0][-21:-4])
creation_date_stop = str(bmp_list[-1][-21:-4])
    


# In[69]:


filename = os.path.basename(filepath)
num_data = len(bmp_list)


print(filename)
print(num_data)
print(creation_date_start)
print(creation_date_stop)
print(notes)


# In[70]:


csv_path = "/Users/anh5866/Desktop/Coding/OGRE/collimated_beam/collimated_beam_entry.xlsx"


# In[71]:


log_path = (nested_nested_folder_path / filename).__str__() + ".txt"

with open(log_path, "w") as f:
    f.write("Filename\tCreation Date\n")
    f.write("-----------------------------------\n")

    for bmp_file in bmp_list:
        #get timestamp 
        dt_str = bmp_file[-21:-4]   
        
        #convert timestamp string to  datetime object
        creation_date = datetime.strptime(dt_str, "%y-%m-%d %H-%M-%S")

        #writee filename and timestamp
        f.write(f"{os.path.basename(bmp_file)}\t{creation_date}\n")

print(f"Creation dates saved to {log_path}")



# In[72]:


input_folder = filepath
output_folder = nested_nested_folder_path / (foldername + "_fits")

print(input_folder)
print(output_folder)


# In[73]:


os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".bmp"):
        #load bmp
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("L")  
        data = np.array(img)

        #save as FITS
        fits_filename = os.path.splitext(filename)[0] + ".fits"
        fits_path = os.path.join(output_folder, fits_filename)
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(fits_path, overwrite=True)



# In[74]:


path2 = output_folder.__str__() + "/*.fits"

fits_list = glob.glob(path2)
fits_list = np.sort(fits_list,kind='standardsort')

print(fits_list)
print(len(fits_list))


# In[ ]:





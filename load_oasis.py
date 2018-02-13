import pandas as pd
import os 
import cv2
import numpy as np
from keras.utils import to_categorical

# Collect list of pngs and identify IDs of controls and alzhimer patients
demographics = pd.read_excel('C:/Users/Shak3/OneDrive/Documents/UCLA_BE_188_Meyer/project/oasis_cross-sectional.xls', sheetname=3)
NC = demographics["ID"][demographics["CDR"] == 0] # get control ID
AD = demographics["ID"][demographics["CDR"] != 0] # get alzheimer's ID

# Get demographic variables: format = NC + AD
Sex = demographics["M/F"][demographics["CDR"] == 0].append(demographics["M/F"][demographics["CDR"] != 0])
Age = demographics["Age"][demographics["CDR"] == 0].append(demographics["Age"][demographics["CDR"] != 0])

# Define outcome: 0 = NC; 1 = AD
y_ohe = to_categorical([0]*NC.shape[0] + [1]*AD.shape[0]) # one-hot encoded outcome
y_bin = [0]*NC.shape[0] + [1]*AD.shape[0] # binary outcome

# Define function to get list of pngs
path='C:/Users/Shak3/OneDrive/Documents/UCLA_BE_188_Meyer/project/OASIS_pngs'
def getPNG(path):
    l = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames: 
            l.append(os.path.join(root,filename))
    return l

# find files and only keep 90th slice
f = getPNG(path)
imgs_tmp = filter(lambda x:'.90.png' in x, f)

# identify NC and AD images
imgs_nc = [filter(lambda x: y in x, imgs_tmp) for y in NC]
imgs_ad = [filter(lambda x: y in x, imgs_tmp) for y in AD]

# join image lists; this way we know which are NC or AD
imgs = imgs_nc + imgs_ad

# Prepare images
def prepPNG(imgs):
    # initialize list
    l = []
    
    # load image
    for i in imgs:        
        tmp = cv2.IMREAD_GRAYSCALE(" ".join(i)) # load
        tmp = cv2.resize(np.array(tmp), (224,224))
        l.append(tmp)
    
    return l

images = np.array(prepPNG(imgs))
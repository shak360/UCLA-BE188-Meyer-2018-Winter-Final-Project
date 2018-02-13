
# coding: utf-8

# In[1]:

import pandas as pd
import os 
import cv2
import numpy as np


# In[2]:

demographics = pd.read_excel('oasis_cross-sectional.xls', sheetname=3) # load data
df = demographics.dropna(how='any') # remove NaN values
df_columns = list(demographics.columns)
X_columns = np.delete(df_columns, [6,7], None) # X matrix won't have MMSE or CDR scores
Xdf = df.reindex(columns=X_columns)
X = Xdf.values # creating X


# In[3]:

# Define function to get list of pngs based on slice number
pngs_path='OASIS_MR1_pngs'
def getaxialPNG(path):
    l = []
    axialslice90_files = []
    axialslice91_files = []
    axialslice92_files = []
    axialslice93_files = []
    axialslice94_files = []
    axialslice95_files = []
    axialslice96_files = []
    axialslice97_files = []
    axialslice98_files = []
    axialslice99_files = []
    
    for root, directories, filenames in os.walk(path):
        
        for filename in filenames:
            if ".90." in filename: 
                axialslice90_files.append(os.path.join(root, filename))
            if ".91." in filename: 
                axialslice91_files.append(os.path.join(root, filename))
            if ".92." in filename: 
                axialslice92_files.append(os.path.join(root, filename))
            if ".93." in filename: 
                axialslice93_files.append(os.path.join(root, filename))
            if ".94." in filename: 
                axialslice94_files.append(os.path.join(root, filename))
            if ".95." in filename: 
                axialslice95_files.append(os.path.join(root, filename))
            if ".96." in filename: 
                axialslice96_files.append(os.path.join(root, filename))
            if ".97." in filename: 
                axialslice97_files.append(os.path.join(root, filename))
            if ".98." in filename: 
                axialslice98_files.append(os.path.join(root, filename))
            if ".99." in filename: 
                axialslice99_files.append(os.path.join(root, filename))

    l = list(zip(axialslice90_files, axialslice91_files, axialslice92_files, axialslice93_files, axialslice94_files, axialslice95_files, axialslice96_files, axialslice97_files, axialslice98_files, axialslice99_files))

    return ((np.asarray(l)))

axial_files0 = getaxialPNG(pngs_path)
axial_X_files = np.take(axial_files0, indices=df.index.values, axis=0) # keeps the images with the same index as X matrix
axial90loc, axial91loc, axial92loc, axial93loc, axial94loc, axial95loc, axial96loc, axial97loc, axial98loc, axial99loc = zip(*axial_X_files)


# In[4]:

X_id, X_sex, X_handedness, X_age, X_education, X_SES, X_eTIV, X_nWBV, X_ASF = zip(*X) # unzips big X matrix

def sex_translator(X_sex):
    X_sex_binary = []
    X_sex_encoded = []
    for x in X_sex:
        if x == 'M':
            X_sex_binary.append(1)
            X_sex_encoded.append([0,1])
        else:
            X_sex_binary.append(-1)
            X_sex_encoded.append([1,0])
    
    return(zip(X_sex_binary, X_sex_encoded)) # gives us binary and one-hot encoded for sex
           
def hand_translator(X_handedness):
    X_hand_binary = []
    X_hand_encoded = []
    for x in X_handedness:
        if x == 'R':
            X_hand_binary.append(1)
            X_hand_encoded.append([0,1])
        else:
            X_hand_binary.append(-1)
            X_hand_encoded.append([1,0])
    
    return(zip(X_hand_binary, X_hand_encoded)) # same as above but for handedness

# turns out all the patients are right-handed
# maybe we should all just be left-handed so we don't suffer from Alzheimer's
# /frequentist sarcasm

X_sex_binary, X_sex_encoded = zip(*sex_translator(X_sex)) # unzipping to get our function outputs
X_hand_binary, X_hand_encoded = zip(*hand_translator(X_handedness))


# In[5]:

def prepPNGimgs(array_of_image_paths):
    l = []
    for img_file in array_of_image_paths: # for each file in the list of images...
        img = cv2.imread("{}".format(img_file)) # read the image...
        img = cv2.resize(np.array(img), (224,224)) # resize it to 224 by 224 QUICK WHAT'S 224 SQUARED??
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # makes it grayscale
        flatten = gray.flatten() # spaghettifies it into 1 by 50176 (which is 224 squared)
        l.append(flatten)
    
    return(np.asarray(l))


# In[6]:

Y_CDR_columns = [column_name for column_name in df_columns if column_name == 'CDR']
Y_CDR_df = df.reindex(columns=Y_CDR_columns)
Y_CDR = Y_CDR_df.values

Y_MMSE_columns = [column_name for column_name in df_columns if column_name == 'MMSE']
Y_MMSE_df = df.reindex(columns=Y_MMSE_columns)
Y_MMSE = Y_MMSE_df.values

CDR_threshold_0 = 0 # threshold values by CDR scale
CDR_threshold_0point5 = 0.5
CDR_threshold_1 = 1

MMSE_threshold_24 = 24 # threshold values by MMSE scale
MMSE_threshold_18 = 18

def CDR_probable_AD_thresholder(Y_CDR, threshold_value):
    Y_CDR_binary = []
    Y_CDR_encoded = []
    for y in Y_CDR:
        if y > threshold_value:
            Y_CDR_binary.append(1)
            Y_CDR_encoded.append([0,1])
        else:
            Y_CDR_binary.append(-1)
            Y_CDR_encoded.append([1,0])

    return((zip(Y_CDR_binary, Y_CDR_encoded)))

def MMSE_probable_Dementia_thresholder(Y_MMSE, threshold_value):
    Y_MMSE_binary = []
    Y_MMSE_encoded = []
    for y in Y_MMSE:
        if y < threshold_value:
            Y_MMSE_binary.append(1)
            Y_MMSE_encoded.append([0,1])
        else:
            Y_MMSE_binary.append(-1)
            Y_MMSE_encoded.append([1,0])
        
    return(zip(Y_MMSE_binary, Y_MMSE_encoded))

Y_CDR_binary, Y_CDR_encoded = zip(*CDR_probable_AD_thresholder(Y_CDR, CDR_threshold_0))
Y_MMSE_binary, Y_MMSE_encoded = zip(*MMSE_probable_Dementia_thresholder(Y_MMSE, MMSE_threshold_24))


# In[7]:

# turning everything into numpy arrays because why not

df_index = np.asarray(df.index.values)
X_id = np.asarray(X_id)
X_sex = np.asarray(X_sex)
X_sex_binary = np.asarray(X_sex_binary)
X_sex_encoded = np.asarray(X_sex_encoded)
X_handedness = np.asarray(X_handedness)
X_hand_binary = np.asarray(X_hand_binary)
X_hand_encoded = np.asarray(X_hand_encoded)
X_age = np.asarray(X_age) 
X_education = np.asarray(X_education) 
X_SES = np.asarray(X_SES)
X_eTIV = np.asarray(X_eTIV)
X_nWBV = np.asarray(X_nWBV)
X_ASF = np.asarray(X_ASF)
axial90loc = np.asarray(axial90loc) 
axial91loc = np.asarray(axial91loc)
axial92loc = np.asarray(axial92loc) 
axial93loc = np.asarray(axial93loc)
axial94loc = np.asarray(axial94loc)
axial95loc = np.asarray(axial95loc) 
axial96loc = np.asarray(axial96loc) 
axial97loc = np.asarray(axial97loc) 
axial98loc = np.asarray(axial98loc) 
axial99loc = np.asarray(axial99loc)
axial90_spaghetti = prepPNGimgs(axial90loc)
axial91_spaghetti = prepPNGimgs(axial91loc)
axial92_spaghetti = prepPNGimgs(axial92loc)
axial93_spaghetti = prepPNGimgs(axial93loc)
axial94_spaghetti = prepPNGimgs(axial94loc)
axial95_spaghetti = prepPNGimgs(axial95loc)
axial96_spaghetti = prepPNGimgs(axial96loc)
axial97_spaghetti = prepPNGimgs(axial97loc)
axial98_spaghetti = prepPNGimgs(axial98loc)
axial99_spaghetti = prepPNGimgs(axial99loc)
Y_CDR = np.squeeze(np.asarray(Y_CDR))
Y_CDR_binary = np.asarray(Y_CDR_binary)
Y_CDR_encoded = np.asarray(Y_CDR_encoded)
Y_MMSE = np.squeeze(np.asarray(Y_MMSE))
Y_MMSE_binary = np.asarray(Y_MMSE_binary)
Y_MMSE_encoded = np.asarray(Y_MMSE_encoded)


# In[9]:

FinalX = np.vstack((X_id, X_sex, X_sex_binary, X_handedness, X_hand_binary, X_age, X_education, X_SES, X_eTIV, X_nWBV, X_ASF))
FinalX = FinalX.T

FinalY = np.vstack((Y_CDR, Y_CDR_binary, Y_MMSE, Y_MMSE_binary))
FinalY = FinalY.T

# concatenating and then turning them into dataframes then into CSVs

XDATAFRAME = pd.DataFrame((FinalX), index=df_index, columns=['X_id', 'X_sex', 'X_sex_binary', 'X_handedness', 'X_hand_binary', 'X_age', 'X_education', 'X_SES', 'X_eTIV', 'X_nWBV', 'X_ASF'])

YDATAFRAME = pd.DataFrame((FinalY), index=df_index, columns=['Y_CDR', 'Y_CDR_binary', 'Y_MMSE', 'Y_MMSE_binary'])

axial90s = pd.DataFrame((axial90_spaghetti), index=df_index)
ouput = pd.DataFrame((Y_CDR_encoded), index=df_index, columns=['NC','AD'])

NeuralNetDF = pd.concat([axial90s, ouput], axis=1)

XDATAFRAME.to_csv('X_DataFrame.csv')
YDATAFRAME.to_csv('Y_DataFrame.csv')
NeuralNetDF.to_csv('NN_Data_frame_90.csv') 

# good luck - sv


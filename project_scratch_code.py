import numpy as np 
import nilearn as nl
from nilearn import datasets
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt 

n_subjects = 25

oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects, dartel_version=False, verbose=1)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
age = oasis_dataset.ext_vars['age'].astype(float)

print('first gray-matter anatomy image in 3D is located at: %s' %oasis_dataset.gray_matter_maps[0])
print('first white-matter anatomy image in 3D is located at: %s' %oasis_dataset.white_matter_maps[0])

for i in oasis_dataset:
 print(oasis_dataset[i])


nlplt.plot_img('C:/Users\Shak3/nilearn_data\oasis1\OAS1_0001_MR1\mwc1OAS1_0001_MR1_mpr_anon_fslswapdim_bet.nii.gz', colorbar= True)

nlplt.plot_img('C:/Users\Shak3/nilearn_data\oasis1\OAS1_0001_MR1\mwc2OAS1_0001_MR1_mpr_anon_fslswapdim_bet.nii.gz', colorbar= True)
nlplt.show()
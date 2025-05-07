import numpy as np
import glob
import os
import time
from tqdm.auto import tqdm
from nilearn.image import resample_img

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

# Hardcode language to French
lang = 'fr'

target_affine = np.array([[   4.,    0.,    0.,  -72.],
                          [   0.,    4.,    0., -106.],
                          [   0.,    0.,    4.,  -64.],
                          [   0.,    0.,    0.,    1.]])

target_shape = (37, 46, 38)

output_path = os.path.join(lbl.home_folder, f'lpp_{lang}_resampled')

print(f'Resampling fMRI data for {lang}...')
subject_list = np.sort(glob.glob(os.path.join(lbl.fmri_data, f'sub-{lang.upper()}*')))
print(f'Found {len(subject_list)} subjects')

start_time = time.time()
for sub_id in tqdm(subject_list):
    sub_name = os.path.basename(sub_id)
    print(f'Processing subject: {sub_name}')   
    make_dir(os.path.join(output_path, sub_name))

    fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, 'func/*.nii.gz')))
    print(f'Found {len(fmri_imgs_sub)} runs')
    for run, fmri_imgs_sub_run in enumerate(fmri_imgs_sub):
        img_resampled = resample_img(fmri_imgs_sub_run, 
                                     target_affine=target_affine, 
                                     target_shape=target_shape)
        img_resampled.to_filename(os.path.join(output_path,
                                               sub_name,
                                               f'{sub_name}_run{run+1}.nii.gz'))

end_time = time.time()
print(f'Resampling completed in {end_time - start_time:.2f} seconds')
print(f'Resampled fMRI data saved in {output_path}')
print('All done!')
import numpy as np
import glob
import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.masking import compute_multi_epi_mask, intersect_masks
from nilearn.image import swap_img_hemispheres
import time

import llms_brain_lateralization as lbl

# Hardcoded language
lang = 'fr'

start_time = time.time()
print(f'Computing mask for {lang}...')

# Path to resampled French fMRI data
fmri_data_resampled = os.path.join(lbl.home_folder, f'lpp_{lang}_resampled')
subject_list = np.sort(glob.glob(os.path.join(fmri_data_resampled, f'sub-{lang.upper()}*')))

# Gather all resampled fMRI runs
fmri_imgs_subs = []
for sub_id in subject_list:
    print(f'Processing {sub_id}')
    fmri_imgs_subs.append(sorted(glob.glob(os.path.join(sub_id, '*.nii.gz'))))

# Compute brain mask from all runs (voxels consistently active)
mask = compute_multi_epi_mask(np.concatenate(fmri_imgs_subs), threshold=0.5)

# Plot the original mask
plotting.plot_roi(mask, title=f'Mask for {lang} language',
                  display_mode='ortho', cut_coords=(0, 0, 0), colorbar=True)
plt.savefig(f'mask_{lang}.png')
plt.close()

# Make the mask symmetrical between hemispheres
mask_sym = intersect_masks([mask, swap_img_hemispheres(mask)], threshold=1)

# Save final mask
nib.save(mask_sym, f'mask_lpp_{lang}.nii.gz')

end_time = time.time()
print(f'Mask computation took {end_time - start_time:.2f} seconds')
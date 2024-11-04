import numpy as np
import glob
import os
import argparse
import nibabel as nib

import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.masking import compute_multi_epi_mask, intersect_masks
from nilearn.image import swap_img_hemispheres

import llms_brain_lateralization as lbl

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='en',
                    help='language: en, fr or cn')
args = parser.parse_args()
lang = args.lang.lower()

assert lang in ['en', 'fr', 'cn'], 'This language is not available. Please choose between en, fr or cn.'

fmri_data_resampled = os.path.join(lbl.home_folder, 'lpp_{}_resampled'.format(lang))
subject_list = np.sort(glob.glob(os.path.join(fmri_data_resampled, 'sub-{}*'.format(lang.upper()))))

fmri_imgs_subs = []
for sub_id in subject_list:
    sub_id_basename = os.path.basename(sub_id)
    fmri_imgs_subs.append(sorted(glob.glob(os.path.join(sub_id, '*.nii.gz'))))

mask = compute_multi_epi_mask(np.concatenate(fmri_imgs_subs), threshold=0.5)

# symmetrize the mask
mask_sym = intersect_masks([mask, swap_img_hemispheres(mask)], threshold=1) 

nib.save(mask_sym, 'mask_lpp_{}.nii.gz'.format(lang))
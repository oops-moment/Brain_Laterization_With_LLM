import pandas as pd
import numpy as np
import os
from joblib import load
from nilearn import plotting
from nilearn.image import iter_img
from nilearn.masking import intersect_masks
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn.maskers import NiftiMasker
import nibabel as nib

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

output_path = lbl.roi_masks
make_dir(output_path)

radius = 10

roi_dict = {'TP':(-48, 15, -27), # from Pallier et al (2011)
            'aSTS':(-54, -12, -12), # from Pallier et al (2011)
            'pSTS':(-51, -39, 3), # from Pallier et al (2011)
            'BA45':(-52, 28, 10), # from Zaccarella et al (2017)
            'BA47':(-44, 34, -8), # from Zaccarella et al (2017)
            'BA44':(-50, 12, 16), # from Zaccarella et al (2017)          
            'AG_TPJ':(-52, -56, 22), # from Price et al (2015)
           }

roi_coords = roi_dict.values()
roi_names = roi_dict.keys()

def binarize_img(img, threshold):
    mask = img.get_fdata().copy()
    mask[mask < threshold] = 0.
    mask[mask >= threshold] = 1.
    return nib.Nifti1Image(mask, img.affine)

def create_roi_from_coords(coords, mask_img=None, radius=10):
    if mask_img is None:
        from nilearn.datasets import load_mni152_brain_mask
        mask_img = load_mni152_brain_mask()
    _, A =  _apply_mask_and_get_affinity(coords, mask_img, radius, True, mask_img=mask_img)
    nifti_masker = NiftiMasker(mask_img=mask_img)
    nifti_masker.fit()
    roi_masks = binarize_img(nifti_masker.inverse_transform(A.toarray()), 0.5)
    return roi_masks

mask_img = nib.load('mask_lpp_en.nii.gz')
roi_masks = create_roi_from_coords(roi_coords, mask_img=mask_img, radius=radius)

for roi_mask, roi_name in zip(iter_img(roi_masks), roi_names):
    filename = os.path.join(output_path, roi_name+'.nii.gz')
    nib.save(roi_mask, filename)
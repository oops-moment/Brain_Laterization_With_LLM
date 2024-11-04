import numpy as np
import pandas as pd
import os
import glob
import joblib
import time
from sklearn.linear_model import Ridge
from nilearn.glm.first_level import compute_regressor
import argparse
import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir, standardize
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_multi_epi_mask, intersect_masks
from nilearn.image import swap_img_hemispheres
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2',
                    help='model name')
# parser.add_argument('--lang', type=str, default='en',
#                     help='language: en, fr or cn')
parser.add_argument('--subject', type=str, default='EN057',
                    help='model name')
args = parser.parse_args()

model_name = args.model
sub_id = args.subject
lang = sub_id[0:2].lower()

assert lang in ['en', 'fr', 'cn'], 'This language is not available. Please choose between en, fr or cn.'

activation_folder = lbl.llms_activations
output_folder = lbl.llms_brain_correlations_individual
make_dir(output_folder)

fmri_data_resampled = os.path.join(lbl.home_folder, 'lpp_{}_resampled'.format(lang))
sub_id = os.path.join(fmri_data_resampled, 'sub-{}'.format(sub_id))

n_runs = lbl.n_runs
t_r = lbl.t_r

hrf_model = 'glover'

# fMRI
fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, '*.nii.gz')))

mask_name = 'mask_lpp_{}.nii.gz'.format(sub_id[-5:])
if os.path.exists(mask_name):
    mask = nib.load(mask_name)
else:
    print('\n computing mask for subject {}'.format(sub_id[-5:]))
    mask = compute_multi_epi_mask(fmri_imgs_sub, threshold=0.5)
    # symmetrize the mask
    mask = intersect_masks([mask, swap_img_hemispheres(mask)], threshold=1.)
    nib.save(mask, mask_name)

fmri_runs = [] # n_runs x n_timesteps x n_voxels
for fmri_img in fmri_imgs_sub:
    nifti_masker = NiftiMasker(mask_img=mask, detrend=True, standardize=True,
                                smoothing_fwhm=4, high_pass=1/128, t_r=t_r)
    fmri_runs.append(nifti_masker.fit_transform(fmri_img))

# number of scans per runs
n_scans_runs = [fmri_run.shape[0] for fmri_run in fmri_runs]

n_voxels = fmri_runs[0].shape[1]

# trim first 20 first seconds, ie 10 first elements with a tr of 2s
# same for last 20 seconds
for run in range(n_runs):
    fmri_runs[run] = fmri_runs[run][10:-10] 
    fmri_runs[run] = standardize(fmri_runs[run])    

print('\n fitting model {}'.format(model_name))

# LLM
filename = os.path.join(activation_folder, '{}_{}.gz'.format(model_name, lang))
with open(filename, 'rb') as f:
    activations_runs_layers_words_neurons = joblib.load(f)
    
# corresponding onsets/offsets    
filename = os.path.join(activation_folder, 'onsets_offsets_{}.gz'.format(lang))
with open(filename, 'rb') as f:
    runs_onsets_offsets = joblib.load(f)

runs_onsets = []
runs_offsets = []

for run in range(n_runs):
    runs_onsets.append(runs_onsets_offsets[run][0])
    runs_offsets.append(runs_onsets_offsets[run][1])

n_layers = len(activations_runs_layers_words_neurons[0])

def compute_regressor_from_activations(activations, onsets, offsets, frame_times):
    # activations: n_timesteps x n_neurons
    durations = offsets - onsets
    nn_signals = []
    for amplitudes in activations.T:
        exp_condition = np.array((onsets, durations, amplitudes))
        signal, name = compute_regressor(
                    exp_condition, hrf_model, frame_times)
        nn_signals.append(signal[:,0])
    return np.array(nn_signals).T # n_scans x n_neurons

alphas = np.logspace(2,7,16)

for idx_layer in range(n_layers):
    print('='*62)
    print('layer {}'.format(idx_layer))
    regressors_runs = []
    for run in range(n_runs): 
        activations_words_neurons = np.array(activations_runs_layers_words_neurons[run][idx_layer]) # words x n_neurons
        onsets = runs_onsets[run]
        offsets = runs_offsets[run]
        frame_times = np.arange(n_scans_runs[run])*t_r  + .5*t_r
        regresssor_run = compute_regressor_from_activations(activations_words_neurons, onsets, offsets, frame_times)
        regresssor_run = regresssor_run[10:-10] #trim
        regresssor_run = standardize(regresssor_run)
        regressors_runs.append(regresssor_run)

    corr_runs = []
    coef_runs = []
    for run_test in range(n_runs):
        tic = time.time()
        
        runs_train = np.setdiff1d(np.arange(n_runs), run_test)
        x_train = np.vstack([regressors_runs[run_train] for run_train in runs_train])
        x_test = regressors_runs[run_test]
        y_train = np.vstack([fmri_runs[run_train] for run_train in runs_train])
        y_test = fmri_runs[run_test]
        
        ############ start nested CV 
        # leave another run apart as a validation test
        run_val = runs_train[0]
        runs_train_val = np.setdiff1d(runs_train, run_val)
        x_train_val = np.vstack([regressors_runs[run_train_val] for run_train_val in runs_train_val])
        x_val = regressors_runs[run_val]
        y_train_val = np.vstack([fmri_runs[run_train] for run_train in runs_train_val])
        y_val = fmri_runs[run_val]

        corr_val = []
        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(x_train_val, y_train_val)
            y_pred = model.predict(x_val)
            corr_tmp = [np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]
            corr_val.append(corr_tmp)    
        idx_best_alpha = np.argmax(np.mean(corr_val, axis=1))
        alpha = alphas[idx_best_alpha]
        ############ end nested CV 
        
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        corr_tmp = [np.corrcoef(y_test[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]

        corr_runs.append(corr_tmp)
        
        toc = time.time()
        
        print('run {}'.format(run_test), '\t', 'mean = {:.03f}'.format(np.mean(corr_tmp)), '\t',
            'max = {:.03f}'.format(np.max(corr_tmp)), '\t',
            'time elapsed = {:.03f}'.format(toc-tic))

    print('---->', '\t' 'mean corr = {:.03f}'.format(np.mean(corr_runs)))

    filename = os.path.join(output_folder, '{}_layer-{}_corr_{}.gz'.format(model_name, idx_layer, sub_id[-5:]))
    with open(filename, 'wb') as f:
        joblib.dump(np.mean(corr_runs, axis=0), f, compress=4)
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2',
                    help='gpt2 variants or opt variants')
args = parser.parse_args()

model_name = args.model
  
activation_folder = lbl.llms_activations
output_folder = lbl.llms_brain_correlations
make_dir(output_folder)

print('\n fitting model {}'.format(model_name))

n_runs = lbl.n_runs
t_r = lbl.t_r

hrf_model = 'glover'

# fMRI
fmri_runs = []
for run in range(n_runs):
    filename = os.path.join(lbl.fmri_data_avg_subject, 'average_subject_run-{}.gz'.format(run))
    with open(filename, 'rb') as f:
         fmri_runs.append(joblib.load(f))

# number of scans per runs
n_scans_runs = [fmri_run.shape[0] for fmri_run in fmri_runs]

n_voxels = fmri_runs[0].shape[1]

# trim first 20 first seconds, ie 10 first elements with a tr of 2s
# same for last 20 seconds
for k in range(n_runs):
    fmri_runs[k] = fmri_runs[k][10:-10] 

for run in range(n_runs):
    fmri_runs[run] = standardize(fmri_runs[run])
    
# LLM
filename = os.path.join(activation_folder, '{}.gz'.format(model_name))
with open(filename, 'rb') as f:
    activations_runs_layers_words_neurons = joblib.load(f)
    
# corresponding onsets/offsets    
filename = os.path.join(activation_folder, 'onsets_offsets.gz')
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
        frame_times = np.arange(n_scans_runs[run]) * t_r  + .5*t_r
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

    filename = os.path.join(output_folder, '{}_layer-{}_corr.gz'.format(model_name, idx_layer))
    with open(filename, 'wb') as f:
        joblib.dump(np.mean(corr_runs, axis=0), f, compress=4)
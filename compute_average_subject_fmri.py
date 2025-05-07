import numpy as np
import os
import glob
import joblib
import argparse
import time
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from datetime import datetime

from nilearn.input_data import NiftiMasker
import nibabel as nib

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir, standardize

# Set up logging
log_dir = "logs"
make_dir(log_dir)
log_filename = os.path.join(log_dir, f"fmri_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Create a directory for visualizations
viz_dir = "visualizations"
make_dir(viz_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='en',
                    help='language: en, fr or cn')
args = parser.parse_args()
lang = args.lang.lower()

assert lang in ['en', 'fr', 'cn'], 'This language is not available. Please choose between en, fr or cn.'
logging.info(f"Processing language: {lang}")

fmri_data_resampled = os.path.join(lbl.home_folder, f'lpp_{lang}_resampled')
subject_list = np.sort(glob.glob(os.path.join(fmri_data_resampled, f'sub-{lang.upper()}*')))
logging.info(f"Found {len(subject_list)} subjects: {[os.path.basename(s) for s in subject_list]}")

fmri_data_avg_subject = os.path.join(lbl.home_folder, f'lpp_{lang}_average_subject')
make_dir(fmri_data_avg_subject)
logging.info(f"Output directory for averaged subjects: {fmri_data_avg_subject}")

n_runs = lbl.n_runs
t_r = lbl.t_r
logging.info(f"Number of runs: {n_runs}, TR: {t_r}")

# Load mask
mask_img_path = f'mask_lpp_{lang}.nii.gz'
logging.info(f"Using mask: {mask_img_path}")
mask_img = nib.load(mask_img_path)
mask_data = mask_img.get_fdata()
logging.info(f"Mask shape: {mask_data.shape}, Non-zero voxels: {np.sum(mask_data > 0)}")

# Process each subject and run
logging.info("Starting to process individual subjects...")
fmri_subs_runs = []
run_lengths = []

for sub_id in tqdm(subject_list, desc="Processing subjects"):
    sub_id_basename = os.path.basename(sub_id)
    logging.info(f"Processing subject: {sub_id_basename}")
    
    fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, '*.nii.gz')))
    logging.info(f"Found {len(fmri_imgs_sub)} fMRI images for subject {sub_id_basename}")
    
    fmri_runs = []  # n_runs x n_timesteps x n_voxels
    for i, fmri_img in enumerate(fmri_imgs_sub):
        img_basename = os.path.basename(fmri_img)
        logging.info(f"Processing run {i+1}/{len(fmri_imgs_sub)}: {img_basename}")
        
        nifti_masker = NiftiMasker(mask_img=mask_img_path, detrend=True, standardize=True,
                                  high_pass=1/128, t_r=t_r)
        fmri_data = nifti_masker.fit_transform(fmri_img)
        logging.info(f"Run shape after masking: {fmri_data.shape}")
        run_lengths.append(fmri_data.shape[0])
        
        fmri_runs.append(fmri_data)
        
    fmri_subs_runs.append(fmri_runs)

logging.info(f"All subjects processed. Run lengths: {run_lengths}")
logging.info(f"Average run length: {np.mean(run_lengths):.2f} time points")

# Create a histogram of run lengths
plt.figure(figsize=(10, 6))
plt.hist(run_lengths, bins=10)
plt.title(f'Distribution of Run Lengths for {lang.upper()} Data')
plt.xlabel('Number of Time Points')
plt.ylabel('Frequency')
plt.savefig(os.path.join(viz_dir, f'{lang}_run_lengths_histogram.png'))
plt.close()
logging.info(f"Saved run lengths histogram to {viz_dir}/{lang}_run_lengths_histogram.png")

# Compute average across subjects for each run
logging.info("Computing average across subjects for each run...")
for run in range(n_runs):
    logging.info(f"Computing average for run {run+1}/{n_runs}")
    fmri_mean_sub = np.mean([fmri_sub_runs[run] for fmri_sub_runs in fmri_subs_runs], axis=0)
    logging.info(f"Mean shape before standardization: {fmri_mean_sub.shape}")
    
    # Plot mean activity before standardization
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(fmri_mean_sub, axis=1))
    plt.title(f'Mean fMRI Activity Before Standardization - Run {run+1}')
    plt.xlabel('Time Points')
    plt.ylabel('Mean Activity')
    plt.savefig(os.path.join(viz_dir, f'{lang}_run{run+1}_mean_activity_before_std.png'))
    plt.close()
    
    fmri_mean_sub = standardize(fmri_mean_sub, axis=0)
    logging.info(f"Mean shape after standardization: {fmri_mean_sub.shape}")
    
    # Plot mean activity after standardization
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(fmri_mean_sub, axis=1))
    plt.title(f'Mean fMRI Activity After Standardization - Run {run+1}')
    plt.xlabel('Time Points')
    plt.ylabel('Standardized Activity')
    plt.savefig(os.path.join(viz_dir, f'{lang}_run{run+1}_mean_activity_after_std.png'))
    plt.close()
    
    filename = os.path.join(fmri_data_avg_subject, f'average_subject_run-{run}.gz')
    with open(filename, 'wb') as f:
        joblib.dump(fmri_mean_sub, f, compress=4)
    logging.info(f"Saved average subject data for run {run+1} to {filename}")

# Compute reliable voxels
logging.info("Starting to compute reliable voxels...")
from sklearn.linear_model import Ridge

np.random.seed(1234)
logging.info("Random seed set to 1234")

n_subjects = len(subject_list)
n_voxels = nifti_masker.n_elements_
logging.info(f"Number of subjects: {n_subjects}, Number of voxels: {n_voxels}")

alphas = [100, 1000, 10000, 100000,10000000]
logging.info(f"Ridge regression alphas: {alphas}")

n_trials = 5
logging.info(f"Number of trials: {n_trials}")

# Create arrays to store results
all_corrs = np.zeros((n_trials, n_runs, n_voxels))
best_alphas = np.zeros((n_trials, n_runs))
mean_corrs_per_trial = np.zeros(n_trials)

corr_split = []
for i_trial in range(n_trials):
    logging.info(f"Starting trial {i_trial+1}/{n_trials}")
    print('='*80)
    
    idx_random = np.arange(n_subjects)
    np.random.shuffle(idx_random)
    
    idx_group_1 = idx_random[:n_subjects//2]
    idx_group_2 = idx_random[n_subjects//2:]
    logging.info(f"Group 1 subjects: {idx_group_1}")
    logging.info(f"Group 2 subjects: {idx_group_2}")
    
    regressors_runs = [np.mean([fmri_subs_runs[idx_sub][run][10:-10] for idx_sub in idx_group_1], axis=0)
                       for run in range(n_runs)]
    fmri_runs = [np.mean([fmri_subs_runs[idx_sub][run][10:-10] for idx_sub in idx_group_2], axis=0)
                 for run in range(n_runs)]
    
    logging.info(f"Regressors shape: {[r.shape for r in regressors_runs]}")
    logging.info(f"fMRI runs shape: {[f.shape for f in fmri_runs]}")

    corr_runs = []
    for run_test in range(n_runs):
        tic = time.time()
        logging.info(f"Testing on run {run_test+1}/{n_runs}")
        
        runs_train = np.setdiff1d(np.arange(n_runs), run_test)
        logging.info(f"Training on runs: {runs_train}")
        
        x_train = np.vstack([regressors_runs[run_train] for run_train in runs_train])
        x_test = regressors_runs[run_test]
        y_train = np.vstack([fmri_runs[run_train] for run_train in runs_train])
        y_test = fmri_runs[run_test]
        
        logging.info(f"Training data shapes - X: {x_train.shape}, Y: {y_train.shape}")
        logging.info(f"Testing data shapes - X: {x_test.shape}, Y: {y_test.shape}")
        
        ############ start nested CV 
        logging.info("Starting nested cross-validation to find best alpha")
        run_val = runs_train[0]
        runs_train_val = np.setdiff1d(runs_train, run_val)
        logging.info(f"Validation run: {run_val}, Training runs for validation: {runs_train_val}")
        
        x_train_val = np.vstack([regressors_runs[run_train_val] for run_train_val in runs_train_val])
        x_val = regressors_runs[run_val]
        y_train_val = np.vstack([fmri_runs[run_train] for run_train in runs_train_val])
        y_val = fmri_runs[run_val]
        
        logging.info(f"Validation training data shapes - X: {x_train_val.shape}, Y: {y_train_val.shape}")
        logging.info(f"Validation testing data shapes - X: {x_val.shape}, Y: {y_val.shape}")

        corr_val = []
        alpha_means = []
        
        for alpha in alphas:
            logging.info(f"Trying alpha={alpha}")
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(x_train_val, y_train_val)
            y_pred = model.predict(x_val)
            
            corr_tmp = []
            for i in range(n_voxels):
                std_val = np.std(y_val[:, i])
                std_pred = np.std(y_pred[:, i])
                if std_val == 0 or std_pred == 0:
                    corr_tmp.append(np.nan)
                else:
                    corr_tmp.append(np.corrcoef(y_val[:, i], y_pred[:, i])[0, 1])
            
            mean_corr = np.nanmean(corr_tmp)
            max_corr = np.nanmax(corr_tmp)
            alpha_means.append(mean_corr)
            corr_val.append(corr_tmp)

            logging.info(f"Ridge alpha={alpha:.1e} => Validation mean={mean_corr:.4f}, max={max_corr:.4f}")
        
        # Plot validation correlations for different alphas
        plt.figure(figsize=(10, 6))
        plt.plot(np.log10(alphas), alpha_means, 'o-')
        plt.xlabel('log10(alpha)')
        plt.ylabel('Mean Correlation')
        plt.title(f'Validation Performance for Different Alphas - Trial {i_trial+1}, Run {run_test+1}')
        plt.savefig(os.path.join(viz_dir, f'{lang}_trial{i_trial+1}_run{run_test+1}_alpha_selection.png'))
        plt.close()

        idx_best_alpha = np.argmax(np.nanmean(corr_val, axis=1))
        alpha = alphas[idx_best_alpha]
        best_alphas[i_trial, run_test] = alpha
        logging.info(f"Best alpha: {alpha}")
        ############ end nested CV 
        
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        logging.info(f"Fitted model with best alpha and predicted test data")

        corr_tmp = []
        for i in range(n_voxels):
            std_test = np.std(y_test[:, i])
            std_pred = np.std(y_pred[:, i])
            if std_test == 0 or std_pred == 0:
                corr_tmp.append(np.nan)
            else:
                corr_tmp.append(np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1])

        corr_runs.append(corr_tmp)
        all_corrs[i_trial, run_test] = corr_tmp
        
        # Plot correlation histogram
        plt.figure(figsize=(10, 6))
        plt.hist(corr_tmp, bins=50, range=(-1, 1))
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        plt.title(f'Distribution of Voxel Correlations - Trial {i_trial+1}, Run {run_test+1}')
        plt.savefig(os.path.join(viz_dir, f'{lang}_trial{i_trial+1}_run{run_test+1}_corr_hist.png'))
        plt.close()
        
        toc = time.time()
        mean_corr = np.nanmean(corr_tmp)
        max_corr = np.nanmax(corr_tmp)
        
        logging.info(f'Run {run_test} done — alpha={alpha:.1e}, mean corr={mean_corr:.4f}, max corr={max_corr:.4f}, time={toc-tic:.2f}s')

    mean_corr_trial = np.nanmean([np.nanmean(c) for c in corr_runs])
    mean_corrs_per_trial[i_trial] = mean_corr_trial
    logging.info(f"Trial {i_trial+1} completed. Mean correlation across all runs: {mean_corr_trial:.4f}")
    
    corr_split.append(np.mean(corr_runs, axis=0))

# Plot mean correlation per trial
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_trials+1), mean_corrs_per_trial, 'o-')
plt.xlabel('Trial')
plt.ylabel('Mean Correlation')
plt.title(f'Mean Correlation by Trial - {lang.upper()}')
plt.grid(True)
plt.savefig(os.path.join(viz_dir, f'{lang}_mean_corr_by_trial.png'))
plt.close()

# Plot heatmap of best alphas
plt.figure(figsize=(10, 6))
sns.heatmap(np.log10(best_alphas), cmap='viridis', 
            xticklabels=[f'Run {i+1}' for i in range(n_runs)],
            yticklabels=[f'Trial {i+1}' for i in range(n_trials)])
plt.xlabel('Run')
plt.ylabel('Trial')
plt.title(f'Best Alpha Values (log10) - {lang.upper()}')
plt.savefig(os.path.join(viz_dir, f'{lang}_best_alphas_heatmap.png'))
plt.close()

# Save the results
filename = f'isc_{n_trials}trials_{lang}.gz'
with open(filename, 'wb') as f:
    joblib.dump(np.array(corr_split), f, compress=4)
logging.info(f"Saved final results to {filename}")

# Create summary statistics
mean_corr_overall = np.nanmean(all_corrs)
std_corr_overall = np.nanstd(all_corrs)
max_corr_overall = np.nanmax(all_corrs)

# Write summary to log
logging.info("=" * 50)
logging.info("SUMMARY STATISTICS")
logging.info("=" * 50)
logging.info(f"Language: {lang.upper()}")
logging.info(f"Number of subjects: {n_subjects}")
logging.info(f"Number of voxels: {n_voxels}")
logging.info(f"Number of runs: {n_runs}")
logging.info(f"Number of trials: {n_trials}")
logging.info(f"Overall mean correlation: {mean_corr_overall:.4f} ± {std_corr_overall:.4f}")
logging.info(f"Overall max correlation: {max_corr_overall:.4f}")
logging.info(f"Mean best alpha: {np.mean(best_alphas):.2f}")

# Create a final summary visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
mean_corrs_by_run = np.nanmean(all_corrs, axis=(0, 2))
plt.bar(range(1, n_runs+1), mean_corrs_by_run)
plt.xlabel('Run')
plt.ylabel('Mean Correlation')
plt.title('Mean Correlation by Run')

plt.subplot(2, 2, 2)
plt.bar(range(1, n_trials+1), mean_corrs_per_trial)
plt.xlabel('Trial')
plt.ylabel('Mean Correlation')
plt.title('Mean Correlation by Trial')

plt.subplot(2, 2, 3)
flat_corrs = all_corrs.flatten()
plt.hist(flat_corrs[~np.isnan(flat_corrs)], bins=50, range=(-1, 1))
plt.xlabel('Correlation')
plt.ylabel('Count')
plt.title('Overall Distribution of Correlations')

plt.subplot(2, 2, 4)
sns.heatmap(np.log10(np.mean(best_alphas, axis=0, keepdims=True)), cmap='viridis',
            xticklabels=[f'Run {i+1}' for i in range(n_runs)],
            yticklabels=['Mean Alpha (log10)'])
plt.title('Mean Best Alpha by Run')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, f'{lang}_summary.png'))
plt.close()

logging.info(f"Summary visualization saved to {viz_dir}/{lang}_summary.png")
logging.info("Script completed successfully!")
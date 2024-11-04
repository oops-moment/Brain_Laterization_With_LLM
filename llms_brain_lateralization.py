import os
import numpy as np

# list of folders used in the study
# main folder, containing code and figures
home_folder = '/media/lbg/hdd1/llms_brain_lateralization/'
# path to Le Petit Prince fMRI corpus, downloaded from https://doi.org/10.18112/openneuro.ds003643.v2.0.5
lpp_path = '/media/lbg/hdd1/data/fmri/openneuro/ds003643-download/'

# fmri data
fmri_data = os.path.join(lpp_path, 'derivatives')
# annotations, used for aligning text and speech
annotation_folder = os.path.join(lpp_path, 'annotation')

# location of the GloVe embeddings
glove_embeddings_path = os.path.join(home_folder, 'glove.6B.300d.txt')
# location of activations from the various llms
llms_activations = os.path.join(home_folder, 'llms_activations')
# location of brain correlations for each model, for each layer
llms_brain_correlations = os.path.join(home_folder, 'llms_brain_correlations')
llms_brain_correlations_individual = os.path.join(home_folder, 'llms_brain_correlations_individual')
# nii files for the roi masks
roi_masks = os.path.join(home_folder, 'roi_masks')
# all figures in the paper
figures_folder = os.path.join(home_folder, 'figures')

n_runs = 9
t_r = 2 #s

acces_token = None

# helpers
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def standardize(v, axis=0):
    return (v - np.mean(v, axis=axis, keepdims=True)) / np.std(v, axis=axis, keepdims=True)

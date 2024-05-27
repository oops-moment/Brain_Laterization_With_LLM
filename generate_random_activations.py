import numpy as np
import pandas as pd
import os
import joblib
import argparse

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

output_folder = lbl.llms_activations
make_dir(output_folder)

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='embedding',
                    help='embedding or vector')
parser.add_argument('--n_dims', type=int, default='300',
                    help='number of dimensions')
parser.add_argument('--seed', type=int, default='1',
                    help='seed')
args = parser.parse_args()

random_type = args.type
n_dims = args.n_dims
seed = args.seed

np.random.seed(seed)

model_name = 'random_{}_{}d_seed{}'.format(random_type, n_dims, seed)

filename = os.path.join(lbl.annotation_folder, 'lppEN_word_information.csv')
df_word_onsets = pd.read_csv(filename)

df_word_onsets = df_word_onsets.drop([3919,6775,6781]) 
# 3919: adhoc removal of repeated line with typo
# 6775: mismatch with full text

word_list_runs = []
onsets_offsets_runs = []
for run in range(lbl.n_runs):
    df_word_onsets_run = df_word_onsets[df_word_onsets.section==(run+1)]
    word_list_tmp = df_word_onsets_run.word.to_numpy()
    onsets_tmp = df_word_onsets_run.onset.to_numpy()
    offsets_tmp = df_word_onsets_run.offset.to_numpy()
    
    word_list = []
    onsets = []
    offsets = []
    
    for idx_word, (word, onset, offset) in enumerate(zip(word_list_tmp, onsets_tmp, offsets_tmp)):
        if isinstance(word, str):
            word_list.append(word)
            onsets.append(onset)
            offsets.append(offset)
            
    onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
    word_list_runs.append(word_list)

runs_words_activations = []

if random_type == 'vector':
    for run in range(lbl.n_runs):
        words_activations = [np.random.randn(n_dims) for _ in range(len(word_list_runs[run]))]
        runs_words_activations.append([words_activations])
elif random_type == 'embedding':
    #first create random embeddings for all words
    word_embeddings = {}        
    for run in range(lbl.n_runs):
        for word in word_list_runs[run]:
            if word not in word_embeddings:
                word_embeddings[word] = np.random.rand(n_dims)
    for run in range(lbl.n_runs):
        words_activations = [word_embeddings[word] for word in word_list_runs[run]]
        runs_words_activations.append([words_activations])
else:
    raise Exception('Unknown random type')

# n_runs x 1 x n_words x n_neurons
filename = os.path.join(output_folder, '{}.gz'.format(model_name))
with open(filename, 'wb') as f:
     joblib.dump(runs_words_activations, f, compress=4)

if not os.path.exists(os.path.join(lbl.llms_activations, 'onsets_offsets.gz')):
    filename = os.path.join(output_folder, 'onsets_offsets.gz')
    with open(filename, 'wb') as f:
         joblib.dump(onsets_offsets_runs, f, compress=4)
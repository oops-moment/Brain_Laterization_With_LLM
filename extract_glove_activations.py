import numpy as np
import pandas as pd
import os
import joblib

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

output_folder = lbl.llms_activations
make_dir(output_folder)

model_name = 'glove'

# code from https://spotintelligence.com/2023/11/27/glove-embedding/
# Load GloVe embeddings into a dictionary
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_embeddings(lbl.glove_embeddings_path)
n_dims = glove_embeddings['the'].shape[0]

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

for run in range(lbl.n_runs):
    word_list = word_list_runs[run]
    
    words_activations = []
    
    for word in word_list:
        # a few adh-oc heuristics to deal with some problematic cases, mainly typos
        word = word.lower().replace("'", "").replace(';','')
        if word == 'na\ive':
             word = 'naive'
        if word == 'redfaced':
             word = 'red-faced'
        if word in glove_embeddings:
            words_activations.append(glove_embeddings[word])
        elif word == 'three two five':
            words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['five'])/3)
        elif word == 'three two six':
            words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['six'])/3)
        elif word == 'three two seven':
            words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['seven'])/3)
        elif word == 'three two eight':
            words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['eight'])/3)
        elif word == 'three two nine':
            words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['nine'])/3)
        elif word == 'three three zero':
            words_activations.append((glove_embeddings['three']*2.0+glove_embeddings['zero'])/3)
        else:
            print('unknown word in run {}: {}'.format(run, word))  
            words_activations.append(np.zeros(n_dims))
    runs_words_activations.append([words_activations])

# n_runs x 1 x n_words x n_neurons
filename = os.path.join(output_folder, '{}.gz'.format(model_name))
with open(filename, 'wb') as f:
     joblib.dump(runs_words_activations, f, compress=4)

if not os.path.exists(os.path.join(lbl.llms_activations, 'onsets_offsets.gz')):
    filename = os.path.join(output_folder, 'onsets_offsets.gz')
    with open(filename, 'wb') as f:
         joblib.dump(onsets_offsets_runs, f, compress=4)
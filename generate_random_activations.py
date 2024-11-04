import numpy as np
import pandas as pd
import os
import joblib
import argparse

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='embedding',
                    help='embedding or vector')
parser.add_argument('--n_dims', type=int, default='300',
                    help='number of dimensions')
parser.add_argument('--seed', type=int, default='1',
                    help='seed')
parser.add_argument('--lang', type=str, default='en',
                    help='Language (en, cn, fr)')
args = parser.parse_args()

random_type = args.type
n_dims = args.n_dims
seed = args.seed
lang = args.lang.lower()

np.random.seed(seed)

output_folder = lbl.llms_activations
make_dir(output_folder)

model_name = 'random_{}_{}d_seed{}_{}'.format(random_type, n_dims, seed, lang)

filename = os.path.join(lbl.annotation_folder, lang.upper(), 'lpp{}_word_information.csv'.format(lang.upper()))
df_word_onsets = pd.read_csv(filename)

if lang == 'en':
    df_word_onsets = df_word_onsets.drop([3919,6775,6781]) 
    # 3919: adhoc removal of repeated line with typo
    # 6775: mismatch with full text
elif lang == 'fr':
    df_word_onsets.loc[3332, 'word'] = 'de' #instead of "du"
    df_word_onsets.loc[3379, 'word'] = 'trois' #instead of "quatre"
    df_word_onsets.loc[3405, 'word'] = 'trois' #instead of "quatre"
    df_word_onsets.loc[4587, 'word'] = 'l' #instead of "d"
    df_word_onsets.loc[5325, 'word'] = 'la' #instead of "le"
    df_word_onsets.loc[5326, 'word'] = 'première' #instead of "premier"
    df_word_onsets.loc[5328, 'word'] = 'habitée' #instead of "habité"
    df_word_onsets.loc[11257, 'word'] = 'À' #instead of "Â"
    df_word_onsets.loc[12249, 'word'] = 'il' #instead of "ll"
    # 338: "s'" in "on s'est égaré", but original text is "on est égaré"  
    # 1204, 3333: empty lines
    df_word_onsets = df_word_onsets.drop([338, 1204, 3333]) 
elif lang == 'cn':
    pass
else:
    raise Exception('This language is not valid.')

n_runs = lbl.n_runs

word_list_runs = []
onsets_offsets_runs = []
for run in range(n_runs):
    df_word_onsets_run = df_word_onsets[df_word_onsets.section==(run+1)]
    word_list_tmp = df_word_onsets_run.word.to_numpy()
    onsets_tmp = df_word_onsets_run.onset.to_numpy()
    offsets_tmp = df_word_onsets_run.offset.to_numpy()
    
    word_list = []
    onsets = []
    offsets = []
    
    for idx_word, (word, onset, offset) in enumerate(zip(word_list_tmp, onsets_tmp, offsets_tmp)):
        if isinstance(word, str) and word != ' ':
            word_list.append(word)
            onsets.append(onset)
            offsets.append(offset)
            
    onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
    word_list_runs.append(word_list)

runs_words_activations = []

if random_type == 'vector':
    for run in range(n_runs):
        words_activations = [np.random.randn(n_dims) for _ in range(len(word_list_runs[run]))]
        runs_words_activations.append([words_activations])
elif random_type == 'embedding':
    #first create random embeddings for all words
    word_embeddings = {}        
    for run in range(n_runs):
        for word in word_list_runs[run]:
            if word not in word_embeddings:
                word_embeddings[word] = np.random.rand(n_dims)
    for run in range(n_runs):
        words_activations = [word_embeddings[word] for word in word_list_runs[run]]
        runs_words_activations.append([words_activations])
else:
    raise Exception('Unknown random type')

# n_runs x 1 x n_words x n_neurons
filename = os.path.join(output_folder, '{}.gz'.format(model_name))
with open(filename, 'wb') as f:
     joblib.dump(runs_words_activations, f, compress=5)

if not os.path.exists(os.path.join(output_folder, 'onsets_offsets_{}.gz'.format(lang))):
    filename = os.path.join(output_folder, 'onsets_offsets_{}.gz'.format(lang))
    with open(filename, 'wb') as f:
         joblib.dump(onsets_offsets_runs, f, compress=5)
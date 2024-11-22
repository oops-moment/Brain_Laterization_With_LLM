import numpy as np
import pandas as pd
import os
import zipfile
import joblib
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizerFast, AutoModel

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2',
                    help='model name')
parser.add_argument('--lang', type=str, default='en',
                    help='language: en, fr or cn')
parser.add_argument('--seed', type=int, default=0,
                help='random seed; 0 means the pretrained model')
args = parser.parse_args()

model_name = args.model
lang = args.lang.lower()

seed = args.seed

if seed > 0:
    torch.manual_seed(seed)

assert lang in ['en', 'fr', 'cn'], 'This language is not available. Please choose between en, fr or cn.'

output_folder = lbl.llms_activations
make_dir(output_folder)

full_text_zip = os.path.join(lbl.home_folder, 'lpp_{}_text.zip'.format(lang))

if model_name.endswith('chinese'):
    full_name = 'ckiplab/{}'.format(model_name)
elif model_name.endswith('french'):
    full_name = 'ClassCat/{}'.format(model_name)
elif model_name.startswith(('gpt2')):
    full_name = 'openai-community/{}'.format(model_name)
elif model_name.startswith('opt'):
    full_name = 'facebook/{}'.format(model_name)
elif model_name.startswith('Mistral'):
    full_name = 'mistralai/{}'.format(model_name)
elif model_name.startswith('gemma'):
    full_name = 'google/{}'.format(model_name)
elif model_name.startswith('mamba'):
    full_name = 'state-spaces/{}'.format(model_name)
elif model_name.startswith('stablelm'):
    full_name = 'stabilityai/{}'.format(model_name)
elif model_name.startswith(('Llama')):
    full_name = 'meta-llama/{}'.format(model_name)
elif model_name.startswith('Qwen'): 
    full_name = 'Qwen/{}'.format(model_name)
elif model_name.startswith('pythia'): 
    full_name = 'EleutherAI/{}'.format(model_name)
else:
    raise Exception('This model does not exist.')


if model_name.startswith(('Meta', 'gemma', 'Mistral', 'Llama')): #require authentification
    assert lbl.access_token != None, "You should provide a valid `access_token` in order to be able to access the various models in huggingface that require authentification."
    model = AutoModelForCausalLM.from_pretrained(full_name, output_hidden_states=True, token=lbl.access_token)
    tokenizer = AutoTokenizer.from_pretrained(full_name, token=lbl.access_token)
elif model_name.startswith('pythia'):        
    tokenizer = AutoTokenizer.from_pretrained(full_name.split('_step')[0])
    model = AutoModelForCausalLM.from_pretrained(full_name.split('_step')[0], revision='step{}'.format(full_name.split('_step')[1]),
                                                  output_hidden_states=True)
elif model_name.endswith('chinese'):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained(full_name, output_hidden_states=True)
else:
    model = AutoModelForCausalLM.from_pretrained(full_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(full_name)

n_layers = model.config.num_hidden_layers + 1 # +1 because of the embedding layer
try:
    maxlen = model.config.max_position_embeddings
except AttributeError:
    # ssm: infinite
    maxlen = 32000 # this is infinity here
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

stride = maxlen - 64

if seed > 0:
    # use random initializations
    model = AutoModelForCausalLM.from_config(model.config)

# Each model is fed with the full original text, with full punctuation. 
# In order to align this text and thus the activity of the neural networks
# with what the subjects in the scanner heard, we make use of the time-aligned
# speech segmentation provided in the LPP naturalistic fMRI corpus, available
# in OpenNeuro. This information is given in the file lppEN_word_information.csv.
# Due to some discrepancies between these two sources, some ad-hoc corrections
# are used in order to improve the alignement.

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

text_archive = zipfile.ZipFile(full_text_zip, 'r')

punct = ['-', "'", "’", "“", '«', '»', '—']
chinese_punct = ['，', '《', '》', '。', '：', '“', '”', '；', '？', '！', '…', '、']
def simplify_word(word):
    word = word.lower()
    word = word.replace(' ', '')
    for p in punct:
        word = word.replace(p, '')
    for p in chinese_punct:
        word = word.replace(p, '')
    return word

def do_word_match(word_in_list, word_in_text):
    word_in_text = simplify_word(word_in_text)
    if len(word_in_text)>0 and word_in_list.startswith(word_in_text):
        return True
    if len(word_in_list)>(1 - (lang=='cn')) and word_in_list in word_in_text:
        return True
    if len(word_in_text)>(1 - (lang=='cn')) and word_in_text in word_in_list:
        return True
    if word_in_list == 'one' and word_in_text == '1':
        return True
    if word_in_list == 'did' and word_in_text == 'didn':
        return True
    if word_in_list == 'nt' and (word_in_text == 't'):
        return True
    if word_in_list == 'does' and word_in_text == "doesn":
        return True
    if word_in_list == 'do' and word_in_text == "don":
        return True
    if word_in_list == 'is' and word_in_text == "isn":
        return True
    if word_in_list == 'threetwofive' and word_in_text == "3":
        return True
    if word_in_list == 'threetwosix' and word_in_text == "3":
        return True
    if word_in_list == 'threetwoseven' and word_in_text == "3":
        return True
    if word_in_list == 'threetwoeight' and word_in_text == "3":
        return True
    if word_in_list == 'threetwonine' and word_in_text == "3":
        return True
    if word_in_list == 'threethreezero' and word_in_text == "3":
        return True
    if word_in_list == 'na\ive' and word_in_text == 'naive':
        return True    
    # fr
    if word_in_list == 'repondit' and word_in_text == 'répondit':
        return True  
    if word_in_list == 'oeuvre' and word_in_text == 'œuvre':
        return True   
    if word_in_list == 'oeil' and word_in_text == 'œil':
        return True   
    if word_in_list == 'a' and word_in_text == 'à':
        return True  
    if word_in_list == 'coeur' and word_in_text == 'cœur':
        return True        
    return False

n_runs = lbl.n_runs

runs_layers_words_activations = [] # n_runs x n_layers x n_words x n_neurons 

word_list_runs = []
onsets_offsets_runs = []
if lang == 'cn':
    # due to the differences between the "word"s in the csv file and how the tokenizer works for Chinese
    # we need here to do some ad-hoc adjustements for this language
    char_to_word_idx_runs = [] #for cn only
for run in range(n_runs):
    df_word_onsets_run = df_word_onsets[df_word_onsets.section==(run+1)]
    word_list_tmp = df_word_onsets_run.word.to_numpy()
    onsets_tmp = df_word_onsets_run.onset.to_numpy()
    offsets_tmp = df_word_onsets_run.offset.to_numpy()
    
    word_list = []
    onsets = []
    offsets = []
    if lang == 'cn':
        char_to_word_idx = []
        
    for idx_word, (word, onset, offset) in enumerate(zip(word_list_tmp, onsets_tmp, offsets_tmp)):
        if isinstance(word, str) and word != ' ':
            if lang == 'cn':
                for char in word:
                    char_to_word_idx.append(idx_word)
                    word_list.append(char)
            else:
                word_list.append(word)
            onsets.append(onset)
            offsets.append(offset)
            
    onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
    word_list_runs.append(word_list)
    if lang == 'cn':
        char_to_word_idx_runs.append(char_to_word_idx)

for run in tqdm(range(n_runs)):
    word_list = word_list_runs[run]

    filename = os.path.join('lpp_{}_text'.format(lang), 'text_{}_run{}.txt'.format(lang, run+1))
    fulltext_run = text_archive.read(filename, pwd=b'lessentielestinvisiblepourlesyeux').decode('utf8')     
    fulltext_run.replace('\n', ' ')
    
    inputs = tokenizer(fulltext_run, 
                           return_tensors='pt',
                           return_offsets_mapping=True, 
                           truncation=True,
                           padding=True,
                           max_length=maxlen,
                           return_overflowing_tokens=True,
                           stride=stride)
    
    # from word in list, find position of corresponding tokens
    idx_batch = 0
    idx_token = 0
    idx_word_to_idx_token = []    
    for idx_word, word in enumerate(word_list):
        word = simplify_word(word)        
        if idx_token == maxlen:
            idx_batch += 1
            idx_token = stride         
        i_start, i_stop = inputs['offset_mapping'][idx_batch, idx_token].numpy()
    
        n = 0
        while n < 20 and not do_word_match(word, fulltext_run[i_start:i_stop]):
            idx_token+=1
            if idx_token == maxlen:
                idx_batch += 1
                idx_token = stride
            i_start, i_stop = inputs['offset_mapping'][idx_batch, idx_token].numpy()
            n += 1
        if n == 20:
            raise Exception('Error while parsing text file -- no corresponding token to word')
            
        # word idx_word starts at idx_token in batch idx_batch
        idx_word_to_idx_token.append((idx_batch, idx_token))
        
        if lang != 'cn':
            idx_token+=1
        elif ((idx_word < len(word_list)-1) and
              (not do_word_match(word_list[idx_word+1], fulltext_run[i_start:i_stop]))):
            idx_token+=1
        
    # could feed it directly all the batches,
    # but it explodes my memory 
    batch_size = inputs['input_ids'].shape[0]
    hidden_states = [] # batch_size x n_layers x n_tokens x n_neurons
    for k in range(batch_size):
        with torch.no_grad():
            model_outputs = model(inputs['input_ids'][k:k+1], 
                                attention_mask=inputs['attention_mask'][k:k+1])
        hidden_states.append([model_outputs['hidden_states'][idx_layer][0].numpy()
                            for idx_layer in range(n_layers)])
            
    layers_words_activations = [[] for _ in range(n_layers)] # n_layers x n_words x n_neurons 
    
    for idx_word in range(len(word_list)-1):
        idx_batch, idx_token = idx_word_to_idx_token[idx_word]
        idx_batch_next, idx_token_next = idx_word_to_idx_token[idx_word+1]
        if idx_batch == idx_batch_next:
            emb_layers = [[] for _ in range(n_layers)] # n_layers x n_tokens x n_neurons 
            for i in range(idx_token, idx_token_next):
                for idx_layer in range(n_layers):
                    emb_layers[idx_layer].append(hidden_states[idx_batch][idx_layer][i])
            for idx_layer in range(n_layers):
                layers_words_activations[idx_layer].append(np.mean(emb_layers[idx_layer], axis=0))  
        else: # idx_batch != idx_batch_next
            emb_layers = [[] for _ in range(n_layers)] # n_layers x n_tokens x n_neurons 
            # go to the end of the batch ...
            for i in range(idx_token, np.minimum(maxlen, inputs['input_ids'][idx_batch].shape[0])):
                for idx_layer in range(n_layers):
                    emb_layers[idx_layer].append(hidden_states[idx_batch][idx_layer][i])  
            # ... and start the next one   
            for i in range(stride, idx_token_next):
                for idx_layer in range(n_layers):
                        emb_layers[idx_layer].append(hidden_states[idx_batch_next][idx_layer][i])
            for idx_layer in range(n_layers):
                layers_words_activations[idx_layer].append(np.mean(emb_layers[idx_layer], axis=0))
    
    # now the last word
    idx_word += 1
    idx_batch, idx_token = idx_word_to_idx_token[idx_word]
    emb_layers = [[] for _ in range(n_layers)] # n_layers x n_tokens x n_neurons 
    for i in range(idx_token, np.minimum(maxlen, inputs['input_ids'][idx_batch].shape[0])):
        token = inputs['input_ids'][idx_batch, i]
        if token == tokenizer.eos_token_id:
            break
        for idx_layer in range(n_layers):
            emb_layers[idx_layer].append(hidden_states[idx_batch][idx_layer][i])
    for idx_layer in range(n_layers):
        layers_words_activations[idx_layer].append(np.mean(emb_layers[idx_layer], axis=0))
    
    runs_layers_words_activations.append(layers_words_activations)

# For the Chinese case, back to associating one position in the csv file provided in the openneuro 
# with an activation pattern
if lang == 'cn':
    runs_layers_chars_activations = runs_layers_words_activations.copy()
    runs_layers_words_activations = [] # n_runs x n_layers x n_words x n_neurons 
    for run in range(n_runs):
        char_to_word_idx = char_to_word_idx_runs[run]
        layers_words_activations = []
        n_words = len(onsets_offsets_runs[run][0])
        for idx_layer in range(n_layers):
            layer_chars_activations = runs_layers_chars_activations[run][idx_layer]
    
            layer_words_activations = []
            
            idx_word = 0
            word_activations = []
            for i in np.flatnonzero(np.array(char_to_word_idx)==idx_word):
                if layer_chars_activations[i].size > 1:
                    word_activations.append(layer_chars_activations[i])
            
            layer_words_activations = []
            for idx_word in range(n_words):
                word_activations = []
                for i in np.flatnonzero(np.array(char_to_word_idx)==idx_word):
                    if layer_chars_activations[i].size > 1:
                        word_activations.append(layer_chars_activations[i])
                if len(word_activations) > 0:
                    layer_words_activations.append(np.mean(word_activations, axis=0))
                else:
                    layer_words_activations.append('tmp')
            # for all positions marked 'tmp', copy vector at the following given position
            # start from the end        
            next_activation = layer_words_activations[-1]
            for idx_word in range(n_words)[::-1]:
                if (isinstance(layer_words_activations[idx_word], str) 
                    and layer_words_activations[idx_word] == 'tmp'):
                    layer_words_activations[idx_word] = next_activation
                next_activation = layer_words_activations[idx_word]
    
            layers_words_activations.append(layer_words_activations)
            
        runs_layers_words_activations.append(layers_words_activations)
        

# n_runs x n_layers x n_words x n_neurons
if seed > 0:
    filename = os.path.join(output_folder, '{}_untrained_seed{}_{}.gz'.format(model_name, seed, lang))
else:
    filename = os.path.join(output_folder, '{}_{}.gz'.format(model_name, lang))

with open(filename, 'wb') as f:
     joblib.dump(runs_layers_words_activations, f, compress=4)

if not os.path.exists(os.path.join(output_folder, 'onsets_offsets_{}.gz'.format(lang))):
    filename = os.path.join(output_folder, 'onsets_offsets_{}.gz'.format(lang))
    with open(filename, 'wb') as f:
         joblib.dump(onsets_offsets_runs, f, compress=4)

import os
import pickle
import pyhocon
import numpy as np

from constants import *
# from transformers import *

def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def compute_average_semtype_emb():
    rs, ctx = 0, 0
    semtype2emb = get_semtype_embs()
    for emb in semtype2emb.values():
        rs += emb
        ctx += 1
    return rs / ctx

def prepare_configs(config_name, dataset, modelname,
                    models_dir, split_nb="", verbose=True):#use_gold = False, 
    # Extract the requested config
    if verbose: print('Config {}'.format(config_name), flush=True)
    BASIC_CONF_PATH = join(BASE_PATH, f'configs/{dataset}.conf')
    configs = pyhocon.ConfigFactory.parse_file(BASIC_CONF_PATH)[config_name]
    configs['dataset'] = dataset
    configs['modelname'] = modelname
    configs['split_nb'] = split_nb
    # configs["use_gold"] = use_gold
    if verbose: print(configs, flush=True)

    # Specific configs for each dataset
    if configs['dataset'] == I2B2:
        configs['entity_types'] = I2B2_ENTITY_TYPES
        configs['relation_types'] = I2B2_RELATION_TYPES
        configs['symmetric_relation'] = False
        configs["notEntityIndex"] = I2B2_ENTITY_TYPES.index(NOT_ENTITY)
        configs["notRelationIndex"] = I2B2_RELATION_TYPES.index(NOT_RELATION)
    elif configs['dataset'] == E3C:
        configs['entity_types'] = E3C_ENTITY_TYPES
        configs['relation_types'] = E3C_RELATION_TYPES
        configs['symmetric_relation'] = False
        configs["notEntityIndex"] = E3C_ENTITY_TYPES.index(NOT_ENTITY)
        configs["notRelationIndex"] = E3C_RELATION_TYPES.index(NOT_RELATION)
    
    # save_dir
    BASE_SAVE_PATH = join(BASE_PATH, models_dir) 
    configs['save_dir'] = join(BASE_SAVE_PATH, '{}_{}'.format(configs['dataset'], config_name))
    create_dir_if_not_exist(configs['save_dir'])

    return configs

def tolist(torch_tensor):
    return torch_tensor.cpu().data.numpy().tolist()

def find_majority(k):
    myMap = {}
    maximum = ( '', 0, 0 ) # (occurring element, occurrences)
    for i, n in enumerate(k):
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n],i)

    return maximum[0], maximum[2]

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def flatten(l):
    return [item for sublist in l for item in sublist]

def listRightIndex(alist, value): # returns the index of first value from right
    return len(alist) - alist[-1::-1].index(value) -1

def inverse_mapping(f):
    return f.__class__(map(reversed, f.items()))

def extract_input_masks_from_mask_windows(mask_windows):
    """
    param mask_windows: e.g. [[-3, 1, 1, 1, -2, -3]],
             [-3, -2, 1, 1, -2, -3],
              [-3, -2, 1, -3, -4, -4] ]
    returns:
    input_masks: e.g.[[1,1,1,1,1,1],
                      [1,1,1,1,1,1],
                    [1,1,1,1,0,0]]
    """
    input_masks = []
    for mask_window in mask_windows:
        subtoken_count = listRightIndex(mask_window, -3) + 1 #get index of rightmost -3 (CLS/SEP token id) after which we have PAD tokens only
        input_masks.append([1] * subtoken_count + [0] * (len(mask_window) - subtoken_count))
    input_masks = np.array(input_masks)
    return input_masks

def convert_to_sliding_window(expanded_tokens, sliding_window_size, tokenizer):
    """
    construct sliding windows, allocate tokens and masks into each window
    :param expanded_tokens: e.g. [tokenid1, tokenid2, tokenid3,tokenid4, tokenid5, tokenid6]
    :param sliding_window_size: e.g. 4
    :return: 
    token_windows: [[CLS, tokenid1, tokenid2, tokenid3, 0, SEP]],
             [CLS, 0, tokenid4, tokenid5, 0, SEP],
              [CLS, 0, tokenid6, SEP, PAD, PAD] ]
    mask_windows: [[-3, 1, 1, 1, -2, -3]],
             [-3, -2, 1, 1, -2, -3],
              [-3, -2, 1, -3, -4, -4]]
    """
    CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])
    SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])
    PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])
    expanded_masks = [1] * len(expanded_tokens)
    sliding_windows = construct_sliding_windows(len(expanded_tokens), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expanded_tokens[window_start: window_end]
        original_masks = expanded_masks[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = CLS + original_tokens + SEP + PAD * (sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    return token_windows, mask_windows

def construct_sliding_windows(sequence_length: int, sliding_window_size: int):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows

# Get total number of parameters in a model
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

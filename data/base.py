import os
import math
import torch
# import pyhocon
import numpy as np

from utils import *

class DataInstance:
    def __init__(self, data, id, text, tokenization, notEntityIndex,
                 entities=None, relations=None, max_tokens=512):
        self.data = data
        self.id = id
        self.text = text
        self.tokens = tokenization['tokens']
        self.entities = entities
        self.relations = relations
        self.tokenization = tokenization
        tokenizer = tokenization['tokenizer']

        self.entity2mentions = {}
        if (not entities is None) and len(entities) > 0:
            for e in entities:
                assert(e['label']!=notEntityIndex)
                self.entity2mentions[e['entity_id']] = e
       
        # Build self.all_relations
        all_relations = {}
        if (not relations is None) and len(relations) > 0:
            for relation in relations:
                p1, p2 = relation['participants']
                mentions1 = self.entity2mentions[p1]
                mentions2 = self.entity2mentions[p2]
                loc1 = mentions1['start_token'], mentions1['end_token']
                loc2 = mentions2['start_token'], mentions2['end_token']
                all_relations[(loc1, loc2)] = relation['label']
                # if symmetric_relation: all_relations[(loc2, loc1)] = relation['label']
        self.all_relations = all_relations

        # Build token_windows, mask_windows, and input_masks
        doc_token_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.token_windows, self.mask_windows = \
            convert_to_sliding_window(doc_token_ids, min(max_tokens, len(doc_token_ids)+2), tokenizer)#+2 for cls, sep
        self.input_masks = extract_input_masks_from_mask_windows(self.mask_windows)

        # Convert to Torch Tensor
        self.input_ids = torch.tensor(self.token_windows)
        self.input_masks = torch.tensor(self.input_masks)
        self.mask_windows = torch.tensor(self.mask_windows)

        # Construct self.gold_starts and self.gold_ends
        gold_starts, gold_ends, gold_labels = [], [], []
        for mi in self.entities:
            gold_starts.append(mi['start_token'])
            gold_ends.append(mi['end_token'])
            gold_labels.append(mi['label'])

        self.gold_starts = torch.tensor(gold_starts)
        self.gold_ends = torch.tensor(gold_ends)
        self.gold_labels = torch.tensor(gold_labels)

        # Construct self.example
        self.isstartingtoken = tokenization['isstartingtoken']
        self.example = (self.input_ids, self.input_masks, self.mask_windows,
                        self.gold_starts, self.gold_ends, self.gold_labels,
                        self.isstartingtoken)

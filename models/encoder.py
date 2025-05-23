import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

from constants import *
from transformers import *

class TransformerEncoder(nn.Module):
    def __init__(self, configs, device):
        super(TransformerEncoder, self).__init__()
        self.configs = configs
        self.device = device

        # Transformer Encoder
        self.transformer = AutoModel.from_pretrained(configs['transformer'])
        self.transformer.config.gradient_checkpointing  = configs['gradient_checkpointing']
        self.transformer_hidden_size = self.transformer.config.hidden_size
        self.hidden_size = self.transformer_hidden_size

    def forward(self, input_ids, input_masks, mask_windows,
                num_windows, window_size, is_training,
                context_lengths = [0], token_type_ids = None):
        """
        input_ids: [[CLS, tokenid1, tokenid2, tokenid3, 0, SEP]],
                    [CLS, 0, tokenid4, tokenid5, 0, SEP],
                    [CLS, 0, tokenid6, SEP, PAD, PAD]]
        param mask_windows: e.g. [[-3, 1, 1, 1, -2, -3]],
                                [-3, -2, 1, 1, -2, -3],
                                [-3, -2, 1, -3, -4, -4] ]
        param input_masks: e.g.[[1,1,1,1,1,1],
                                [1,1,1,1,1,1],
                                [1,1,1,1,0,0]]
        """
        self.train() if is_training else self.eval()
        # print(f"input_masks:{input_masks.shape}")#[1, 193] or [2, 512 etc.]
        # print(f"mask_windows:{mask_windows.shape}")#[1, 193] or [2, 512 etc.]
        # print(f"input ids:{input_ids.shape}") # [1, 193] or [2, 512 etc.]
        # for w in range(num_windows):
        #     print(f"input_ids[{w},0]: {input_ids[w,0]}")#all are same cls
        num_contexts = len(context_lengths)
        # num_contexts:1 num_windows:2 window_size:512

        # last hidden states, cls output:This output is usually not a good summary
        #  of the semantic content of the input, you’re often better with averaging
        #  or pooling the sequence of hidden-states for the whole input sequence.
        features, _ = self.transformer(input_ids, input_masks, token_type_ids)[:2]
        # features:torch.Size([2, 512, 768])

        features = features.view(num_contexts, num_windows, -1, self.transformer_hidden_size)
        # reshaped features:torch.Size([1, 2, 512, 768])

        flattened_features = []
        tokenindex_to_clsindices = []
        cls_embs = []
        for i in range(num_contexts):
            _features = features[i, :, :, :]
            # _features 1:torch.Size([2, 512, 768])

            _features = _features[:, context_lengths[i]:, :]#size of context window is full
            # _features 2:torch.Size([2, 512, 768])

            _features = _features[:, : window_size, :]#size of context window is full window
            # _features 3:torch.Size([2, 512, 768])
            flattened_emb, tokenindex_to_clsindex, cls_emb = self.flatten(_features, mask_windows)
            cls_embs.append(cls_emb)
            flattened_features.append(flattened_emb)
            tokenindex_to_clsindices.append(tokenindex_to_clsindex)

        flattened_features = torch.cat(flattened_features).squeeze()
        tokenindex_to_clsindices = torch.cat(tokenindex_to_clsindices).squeeze()
        cls_embs = torch.cat(cls_embs)
        if cls_embs.dim()==1:
            cls_embs = cls_embs[None,:]#.unsqueeze(0)
        # flattened_features:torch.Size([1024, 768])

        return flattened_features, tokenindex_to_clsindices, cls_embs

    def flatten(self, features, mask_windows):
        num_windows, window_size, hidden_size = features.size()
        # features before flattening:torch.Size([2, 512, 768])
        cls_emb = features[:,0,:].squeeze()
        #print(f"cls_features.shape: {cls_emb.shape}")#[768] or [2, 768] etc
        tokenindex_to_clsindex = []
        for i in range(mask_windows.shape[0]):
            window_positions = [i]*features.shape[1]
            tokenindex_to_clsindex.append(window_positions)
        tokenindex_to_clsindex = torch.LongTensor(tokenindex_to_clsindex).to(self.device)
        #print(f"tokenindex_to_clsindex.shape: {tokenindex_to_clsindex.shape}")#[1, 193], 193 tokens in window1
        #print(f"window_size:{window_size}")#193
        tokenindex_to_clsindex =  torch.reshape(tokenindex_to_clsindex, (num_windows * window_size,))
        
        flattened_emb = torch.reshape(features, (num_windows * window_size, hidden_size))
        # features after flattening:torch.Size([1024, 768]) 
        boolean_mask = mask_windows > 0
        boolean_mask = boolean_mask.view([num_windows * window_size])
        # boolean_mask:torch.Size([1024])
        return flattened_emb[boolean_mask].unsqueeze(0), tokenindex_to_clsindex[boolean_mask].unsqueeze(0), cls_emb

from torch_geometric.nn import HANConv
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
from constants import *
from copy import deepcopy

class DocHetroGraph():
    def __init__(self, device=torch.device("cuda")):
        self.device = device

    def _init_data(self, ent_type_list, rel_type_list):
        self._data = HeteroData().to(self.device)
        self.ent_type_dict = {ent_type:i for i, ent_type in enumerate(ent_type_list)}
        self.rel_type_dict = {rel_type:i for i, rel_type in enumerate(rel_type_list)}
        self.ent_counts = {ent_type:0 for ent_type in ent_type_list}
        self.edge_dict = {}

    def add_sentence_types(
        self, 
        type_idxs, 
        sent_x, 
        sent_idxs, 
        ent_type="SENTENCE", 
        rel_type="BELONGS_TO", 
        ent_drop_keys=None, 
        instant_update=False
    ):
        n_sentes = sent_x.size(0)
        self._data[ent_type].x = sent_x
        if n_sentes > 1:
            hetero_rel = (ent_type, "To", ent_type)
            edges = torch.cat((torch.arange(n_sentes-1)[None, :], torch.arange(n_sentes-1)[None, :]+1), dim=0)
            if instant_update:
                self._data[hetero_rel].edge_index = edges.to(self.device)
            else:
                self.edge_dict[hetero_rel] = edges.to(self.device)
            
        ent_type_dict = {key:value for key, value in self.ent_type_dict.items() if key not in ent_drop_keys}
        for etype, j in ent_type_dict.items():
            temp_mask = type_idxs[:, 0] == j
            if temp_mask.sum() > 0:
                hetero_rel = (etype, rel_type, ent_type)
                edges = torch.cat((type_idxs[temp_mask, 1][None, :], sent_idxs[temp_mask][None, :]), dim=0)
                if instant_update:
                    self._data[hetero_rel].edge_index = edges
                else:
                    self.edge_dict[hetero_rel] = edges
        
        self.ent_type_dict[ent_type] = max(list(self.ent_type_dict.values())+[0]) + 1
        self.rel_type_dict[rel_type] = max(list(self.rel_type_dict.values())+[0]) + 1

    def add_dummy_types(
        self, 
        dummy_x, 
        dummy_idxs,
        type_idxs,
        ent_type="DUMMY", 
        rel_types=["BEFORE_DUMMY", "AFTER_DUMMY"], 
        ent_drop_keys=None,
        instant_update=False,
        # max_length=20,
        # reduction="max"
    ):
        # num_spans = type_idxs.size(0)
        # exp_ends = end_idxs[:, None].tile(1, num_spans)
        # exp_starts = start_idxs[None, :].tile(num_spans, 1)
        # diffs = exp_starts - exp_ends
        # in_idxs = ((0 < diffs) & (diffs < max_length)).nonzero()
        
        # lex_idxs = torch.cat((end_idxs[in_idxs[:, 0]][:, None], start_idxs[in_idxs[:, 1]][:, None]), dim=1)
        # x = torch.zeros(lex_idxs.size(0), token_x.size(1))
        # for i, lex_range in enumerate(lex_idxs):
        #     temp_x = token_x[lex_range[0]:lex_range[1]]
        #     if reduction == "max":
        #         x[i] = temp_x.max(dim=0).values
        #     if reduction == "mean":
        #         x[i] = temp_x.mean(dim=0)
        #     if reduction == "sum":
        #         x[i] = temp_x.sum(dim=0)
        if dummy_x.size(0) == 0:
            return
        self._data[ent_type].x = dummy_x

        ent_type_dict = {key:value for key, value in self.ent_type_dict.items() if key not in ent_drop_keys}
        idxs = torch.arange(dummy_idxs.size(0)).to(self.device)
        for i, rel_type in enumerate(rel_types):
            temp_idxs = type_idxs[dummy_idxs[:, i]]
            for etype, j in ent_type_dict.items():
                mask = temp_idxs[:, 0] == j
                if mask.sum() > 0:
                    hetero_rel = (
                        etype, 
                        rel_type,
                        ent_type
                    )
                    edges = torch.cat((temp_idxs[mask, 1][None, :], idxs[mask][None, :]), dim=0)
                    if instant_update:
                        self._data[hetero_rel].edge_index = edges
                    else:
                        self.edge_dict[hetero_rel] = edges
                    
        self.ent_type_dict[ent_type] = max(list(self.ent_type_dict.values())+[0]) + 1
        for rel_type in rel_types:
            self.rel_type_dict[rel_type] = max(list(self.rel_type_dict.values())+[0]) + 1

    def update_data(self):
        for key, value in self.edge_dict.items():
            self._data[key].edge_index = value

    def add_span_types(
        self, 
        span_x, 
        spans_types, 
        rel_types, 
        ent_drop_keys=None, 
        rel_drop_keys=None, 
        instant_update=False
    ):
        type_idxs = torch.ones_like(spans_types) * -1
        # ipdb.set_trace()
        ent_type_dict = dict()
        for etype, i in self.ent_type_dict.items():
            if etype in ent_drop_keys:
                continue
            mask_idxs = (spans_types == i).nonzero().T[0]
            if mask_idxs.size(0) > 0:
                self._data[etype].x = span_x[mask_idxs]
                type_idxs[mask_idxs] = torch.arange(mask_idxs.size(0)).to(self.device)
                ent_type_dict[etype] = i
                self.ent_counts[etype] = mask_idxs.size(0)
        type_idxs = torch.cat((spans_types[:, None], type_idxs[:, None]), dim=1)
        self.ent_type_dict = deepcopy(ent_type_dict)

        # ent_type_dict = {key:value for key, value in self.ent_type_dict.items() if key not in ent_drop_keys}
        rel_type_dict = {key:value for key, value in self.rel_type_dict.items() if key not in rel_drop_keys}
        for rel_type, i in rel_type_dict.items():
            mask = (rel_types == i).nonzero()
            temp_idxs = type_idxs[mask]
            for ent_type1, j in ent_type_dict.items():
                src_mask = temp_idxs[:, 0, 0] == j
                src_masked_idxs = temp_idxs[src_mask]
                for ent_type2, k in ent_type_dict.items():
                    hetero_rel = (
                        ent_type1, 
                        rel_type,
                        ent_type2
                    )
                    tgt_mask = src_masked_idxs[:, 1, 0] == k
                    if tgt_mask.sum() > 0:
                        edges = src_masked_idxs[tgt_mask, :, 1].T
                        assert self._data[ent_type1].x.size(0) >= edges[0, :].max() or self._data[ent_type2].x.size(0) >= edges[1, :].max()
                        if instant_update:
                            self._data[hetero_rel].edge_index = edges
                        else:
                            self.edge_dict[hetero_rel] = edges
        return type_idxs

    def setup_graph(
        self, 
        ent_type_dict,
        rel_type_dict,
        span_x, 
        spans_types, 
        rel_types,
        add_dummy=True,
        add_sent=True,
        sent_x=None,
        dummy_x=None, 
        dummy_idxs=None,
        sent_idxs=None,
        ent_drop_keys=None,
        rel_drop_keys=None,
        dummy_max_length=20,
        instant_update=False
    ):
        self._init_data(ent_type_dict, rel_type_dict)
        t_idxs = self.add_span_types(span_x, spans_types, rel_types, ent_drop_keys=ent_drop_keys, rel_drop_keys=rel_drop_keys)
        if add_dummy:
            self.add_dummy_types(dummy_x, dummy_idxs, t_idxs, ent_drop_keys=ent_drop_keys)
        if add_sent:
            self.add_sentence_types(t_idxs, sent_x, sent_idxs, ent_drop_keys=ent_drop_keys)
        self.update_data()
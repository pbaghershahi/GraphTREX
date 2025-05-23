import torch
import utils
import random
import numpy as np
from utils import *
from constants import *
from models.base import *
from models.encoder import *
from models.helpers import *
from models.graph_utils import *
from itertools import combinations
from torch_geometric.nn import HGTConv
import torch_geometric.transforms as T

OUTPUT_FIELDS = ['starts', 'ends', 'entity_labels', 'relation_labels', 
                 'mention_scores', 'relation_scores']#, 'candidate_embs']

class HGT(nn.Module):
    def __init__(
        self, 
        hetero_metadata, 
        in_channels: int,
        out_channels: int, 
        hidden_channels: int, 
        num_layers: int = 1,
        attn_heads: int=8,
        with_decoder=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = 0.3
        self.with_decoder = with_decoder
        if num_layers > 1:
            self.gnn_layers = nn.ModuleList([HGTConv(in_channels, hidden_channels, heads=attn_heads, metadata=hetero_metadata)])
            num_hid_layer = num_layers - 1 if with_decoder else num_layers - 2
            for _ in range(num_hid_layer):
                self.gnn_layers.append(HGTConv(hidden_channels, hidden_channels, heads=attn_heads, metadata=hetero_metadata))
            if with_decoder:
                self.decoder = torch.nn.Linear(hidden_channels, out_channels)
            else:
                self.decoder = HGTConv(hidden_channels, out_channels, heads=attn_heads, metadata=hetero_metadata)
        else:
            self.gnn_layers = nn.ModuleList([HGTConv(in_channels, out_channels, heads=attn_heads, metadata=hetero_metadata)])

    def forward(self, x_dict, edge_index_dict, out_types=None):
        x = x_dict 
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index_dict)
            if i < len(self.gnn_layers) - 1:
                for key, value in x.items():
                    x[key] = F.relu(value)
                    x[key] = F.dropout(value, p=self.dropout, training=self.training)
        if self.with_decoder:
            for key, value in x.items():
                x[key] = self.decoder(value)
        return x

def get_metadata(configs, sent_type="SENTENCE", dummy_type="DUMMY", sent_rels=["BELONGS_TO"], dummy_rels=["BEFORE_DUMMY", "AFTER_DUMMY"]):
    ent_list = configs['entity_types'] + [sent_type, dummy_type]
    rel_list = []
    for r in configs['relation_types'][1:]:
        for e1 in configs['entity_types'][1:]:
            for e2 in configs['entity_types'][1:]:
                    rel_list.append((e1, r, e2))
    
    for r in sent_rels:
        for e in configs['entity_types'][1:]:
            rel_list.append((e, r, sent_type))
    rel_list.append((sent_type, 'To', sent_type))
    
    for r in dummy_rels:
        for e in configs['entity_types'][1:]:
            rel_list.append((e, r, dummy_type))
    return (ent_list, rel_list)
    
# Main Class
class PredictionHead(nn.Module):
    def __init__(self, configs, device):
        super(PredictionHead, self).__init__()
        self.configs = configs
        self.device = device
        self.span_emb_size = configs['span_emb_size']
        
        self.nb_entity_types = len(configs['entity_types'])
        self.nb_relation_types = len(configs['relation_types']) 
        self.bert_emb_size = configs['bert_emb_size']
        self.pair_embs_size = 2 * self.span_emb_size 
        if self.configs['use_prod']:
            self.pair_embs_size+=self.span_emb_size
        if self.configs['use_minus']:
            self.pair_embs_size+= self.span_emb_size
        if self.configs['use_plus']:
            self.pair_embs_size += self.span_emb_size
        if self.configs['USE_ENTITY_TYPES']:
            self.pair_embs_size += 2
        if self.configs['USE_POOLED_CONTEXT']:
            self.pair_embs_size += self.bert_emb_size 
        
        mention_hidden_sizes = [self.configs['mention_scorer_ffnn_size']] * self.configs['mention_scorer_ffnn_depth']
        mention_scorer_input_size = self.span_emb_size
        relation_scorer_input_size = self.pair_embs_size
       
        self.mention_scorer = FFNNModule(input_size = mention_scorer_input_size,
                                         hidden_sizes = mention_hidden_sizes,
                                         output_size = self.nb_entity_types,
                                         dropout = self.configs['dropout_rate'])
       
        self.mention_loss_fct = nn.CrossEntropyLoss()

        # Relation Extractor
        relation_hidden_sizes = [self.configs['mention_linker_ffnn_size']] * self.configs['mention_linker_ffnn_depth']
        self.relation_scorer = FFNNModule(input_size = relation_scorer_input_size,
                                          hidden_sizes = relation_hidden_sizes,
                                          output_size = self.nb_relation_types,
                                          dropout = self.configs['dropout_rate'])
        self.relation_loss_fct = nn.CrossEntropyLoss()

        if self.configs['USE_GNN']:
            self.gnn_model = HGT(
                get_metadata(self.configs), 
                in_channels=-1, 
                out_channels=self.configs['span_emb_size'], 
                hidden_channels=self.configs['span_emb_size'],
                num_layers=self.configs['gnn_num_layers'],
                attn_heads=self.configs['gnn_attn_heads'],
                with_decoder=self.configs['gnn_with_decoder'],
            )
            self.gnn_model.to(device)
            self.hetero_graph = DocHetroGraph(device)

    def forward(self, candidate_starts, candidate_ends, candidate_embs, 
                token_embs, cls_embs, tokenindex_to_clsindices, 
                mention_labels, relation_labels, is_training, in_ned_pretraining):
        
        mention_scores = self.mention_scorer(candidate_embs)
        if len(mention_scores.size()) == 1: mention_scores = mention_scores.unsqueeze(0)
        mention_loss = self.mention_loss_fct(mention_scores, mention_labels) if is_training else 0
        _, pred_mention_labels = torch.max(mention_scores, 1)
        if is_training and in_ned_pretraining: return mention_loss, {l:[] for l in OUTPUT_FIELDS}
        
        # Extract candidates that do not have not-entity label
        top_candidate_indexes = [ix for ix, l in enumerate(tolist(pred_mention_labels)) if l !=self.configs['notEntityIndex']]
        if len(top_candidate_indexes) == 0:
          nb_mentions = len(candidate_starts)
          top_candidate_indexes = random.sample(list(range(nb_mentions)), 1)
        # print(f"len(top_candidate_indexes):{len(top_candidate_indexes)}")
        top_candidate_indexes = torch.tensor(top_candidate_indexes).to(self.device)
        top_candidate_starts = candidate_starts[top_candidate_indexes]
        top_candidate_ends = candidate_ends[top_candidate_indexes]
        top_mention_scores = torch.index_select(mention_scores, 0, top_candidate_indexes)
        top_mention_labels = pred_mention_labels[top_candidate_indexes]
        top_candidate_embs = torch.index_select(candidate_embs, 0, top_candidate_indexes)
        relation_labels = torch.index_select(relation_labels, 0, top_candidate_indexes)
        relation_labels = torch.index_select(relation_labels, 1, top_candidate_indexes)
        
        top_entity_types = top_mention_labels
        
        # if USE_ENTITY_PROB:
        #     top_entity_types = top_mention_scores.clone().detach()

        context_embs = torch.index_select(cls_embs, 0, 
                                          tokenindex_to_clsindices[top_candidate_starts]) ## if USE_CLS_EMBS:
        if self.configs['USE_POOLED_CONTEXT']:#we're pooling spans and not tokens
            context_embs = (token_embs, top_candidate_starts, top_candidate_ends)

        self.configs["itrs_refine"] = self.configs["itrs_refine"] if self.configs['USE_GNN'] else 1
        for itr in range(self.configs["itrs_refine"]):
            # Compute pair_embs
            pair_embs, dummy_embeds, dummy_idxs = self.get_pair_embs(top_candidate_embs, context_embs, top_entity_types)
    
            # Compute pair_relation_scores and pair_relation_loss
            pair_relation_scores = self.relation_scorer(pair_embs)
            if len(pair_relation_scores.size()) <= 1:
                pair_relation_scores = pair_relation_scores.view(1, 1, self.nb_relation_types)
            pair_relation_score, pair_relation_labels = torch.max(pair_relation_scores, 2)
            pair_relation_probs = torch.softmax(pair_relation_scores, dim=-1)

            if self.configs['USE_GNN']:
                none_rel_idxs = self.configs['notRelationIndex']
                rel_scores, rel_types = pair_relation_probs.max(dim=-1)
                rel_mask = rel_scores < self.configs["pred_threshold"]
                rel_types[rel_mask] = none_rel_idxs
                if (rel_types != none_rel_idxs).sum() < 1 or top_candidate_embs.size(0) < 2:
                    break
                self.hetero_graph.setup_graph(
                    self.configs['entity_types'], 
                    self.configs['relation_types'],
                    top_candidate_embs, 
                    top_mention_labels, 
                    rel_types,
                    add_dummy= self.configs["add_dummy_nodes"],
                    add_sent= self.configs["add_sent_nodes"],
                    sent_x=cls_embs,
                    dummy_x=dummy_embeds,
                    dummy_idxs=dummy_idxs,
                    sent_idxs=tokenindex_to_clsindices[top_candidate_ends],
                    ent_drop_keys=[NOT_ENTITY], 
                    rel_drop_keys=[NOT_RELATION], 
                    instant_update=False
                )
                gnn_out = self.gnn_model(self.hetero_graph._data.x_dict, self.hetero_graph._data.edge_index_dict)
                for key, value in gnn_out.items():
                    mask = (top_mention_labels == self.hetero_graph.ent_type_dict[key])
                    if mask.sum() > 0:
                        top_candidate_embs[mask, :] = value + self.configs["residual_coef"] * top_candidate_embs[mask, :]

        pair_relation_loss = 0
        if self.configs['USE_GNN']:
            n, d = top_candidate_embs.size()
            # print(f"n:{n}")
            src_embs = top_candidate_embs.view(1, n, d).repeat([n, 1, 1])
            target_embs = top_candidate_embs.view(n, 1, d).repeat([1, n, 1])
            pair_embs[:, :, :d] = src_embs
            pair_embs[:, :, d:2*d] = target_embs
            if self.configs['use_prod']:
                prod_embs = src_embs*target_embs
                pair_embs[:, :, 2*d:3*d] = prod_embs
    
            # Compute pair_relation_scores and pair_relation_loss
            pair_relation_scores = self.relation_scorer(pair_embs)
            if len(pair_relation_scores.size()) <= 1:
                pair_relation_scores = pair_relation_scores.view(1, 1, self.nb_relation_types)
            pair_relation_score, pair_relation_labels = torch.max(pair_relation_scores, 2)
            pair_relation_probs = torch.softmax(pair_relation_scores, dim=-1)

        if is_training:
            pair_relation_loss = self.relation_loss_fct(pair_relation_scores.view(-1, self.nb_relation_types), relation_labels.view(-1))
            
        # Compute total_loss
        total_loss = mention_loss + pair_relation_loss

        # Build preds
        preds = {
                 'starts': top_candidate_starts, 'ends': top_candidate_ends,
                 'entity_labels': top_mention_labels, 
                 'relation_labels': pair_relation_labels,
                 'mention_scores': top_mention_scores,
                 'relation_scores': pair_relation_probs,
                 'candidate_embs': top_candidate_embs
                 }
        
        return total_loss, preds
    
    def get_pair_embs(self, top_candidate_embs, context_embs, entity_types):#, top_mention_labels):
        n, d = top_candidate_embs.size()
        features_list = []
        # print(f"n:{n}")
        
        # Compute diff_embs and prod_embs
        src_embs = top_candidate_embs.view(1, n, d).repeat([n, 1, 1])
        target_embs = top_candidate_embs.view(n, 1, d).repeat([1, n, 1])
        dummy_idxs = []
        dummy_embeds = []
        
        # Update features_list
        features_list.append(src_embs)
        features_list.append(target_embs)
        if self.configs['use_prod']:
            prod_embs = src_embs*target_embs
            features_list.append(prod_embs)
        
        if self.configs['use_plus']:
            plus_embs = src_embs + target_embs
            features_list.append(plus_embs)

        if self.configs['use_minus']:
            minus_embs = src_embs-target_embs
            features_list.append(minus_embs)

        if self.configs['USE_POOLED_CONTEXT']:
            
            token_embs, start_ix, end_ix = context_embs
            pooled_embs = []                    
            src_start = torch.flatten(start_ix.view(1, n,1).repeat([n,1,1]))
            src_end = torch.flatten(end_ix.view(1,n,1).repeat([n,1,1]))
            target_start = torch.flatten(start_ix.view(n ,1,1).repeat([1,n,1]))
            target_end = torch.flatten(end_ix.view(n ,1,1).repeat([1,n,1]))
            dummy_mask = torch.ones_like(target_end)
            max_length = int(self.configs["POOLING_MAXL_RATIO"] * token_embs.size(0))
            

            nb_tokens = token_embs.size()[0]
            for i in range(src_start.shape[0]):
                s1 = src_start[i]
                e1 = src_end[i]
                s2 = target_start[i]
                e2 = target_end[i]
                if 1 < s2-e1 <= max_length:
                    mask = [0]*(e1+1)+[1]*(s2-e1-1)+[0]*(nb_tokens-s2)
                elif 1 < s1-e2 <= max_length:
                    mask = [0]*(e2+1)+[1]*(s1-e2-1)+[0]*(nb_tokens-s1)
                else:
                    mask = [0]*nb_tokens
                    dummy_mask[i] = 0
                mask = torch.tensor(mask, dtype = top_candidate_embs.dtype).to(self.device)
                max_pooled_emb = torch.max(torch.mul(token_embs, mask[:, None]), 0)[0]
                pooled_embs.append(max_pooled_emb)

            pooled_embs = torch.stack(pooled_embs).view(n,n,token_embs.shape[-1])
            dummy_mask = dummy_mask.view(n, n)
            dummy_idxs = dummy_mask.nonzero()
            dummy_embeds = pooled_embs[dummy_idxs[:, 0], dummy_idxs[:, 1], :]
            features_list.append(pooled_embs)
            
        if self.configs['USE_ENTITY_TYPES']:
            features_list.append(entity_types.view(1, n, 1).repeat([n, 1, 1]))
            features_list.append(entity_types.view(n, 1, 1).repeat([1, n, 1]))
        # Concatenation
        pair_embs = torch.cat(features_list, 2)
        return pair_embs, dummy_embeds, dummy_idxs
    
class JointModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.configs = configs
        self.nb_entity_types = len(self.configs['entity_types'])
        self.nb_relation_types = len(self.configs['relation_types'])
        self.in_ned_pretraining = False

        # Transformer Encoder
        self.encoder = TransformerEncoder(self.configs, self.device)

        # Span Embedding Linear
        self.span_linear_1 = nn.Linear(self.get_span_emb_inp_size(),self.configs['span_emb_size'])
        self.span_relu = nn.ReLU()

        # Span-width features
        self.span_width_embeddings = nn.Embedding(self.configs['max_span_width'], self.configs['feature_size'])
        # Prediction Heads
        self.predictor= PredictionHead(self.configs, self.device)

        self.to(self.device)

    def get_span_emb_inp_size(self):
        sp_emb_inp_size = 2 * self.encoder.hidden_size + self.configs['feature_size']
        return sp_emb_inp_size
                                                                                                
    def forward(self, input_ids, input_masks, mask_windows,
                gold_starts, gold_ends, gold_labels, isstartingtoken,
                relations, data, is_training):
        self.train() if is_training else self.eval()

        num_windows, window_size = input_ids.size()[:2]
        transformer_features, tokenindex_to_clsindices, cls_embs = \
            self.encoder(input_ids, input_masks, mask_windows, num_windows, window_size, is_training)

        num_tokens = transformer_features.size()[0]
        # Enumerate span candidates
        candidate_spans = self.enumerate_candidate_spans(num_tokens, self.configs['max_span_width'], isstartingtoken)
        candidate_spans = sorted(candidate_spans, key=lambda x: x[0])
        candidate_starts = torch.LongTensor([s[0] for s in candidate_spans]).to(self.device)
        candidate_ends = torch.LongTensor([s[1] for s in candidate_spans]).to(self.device)

        # Extract candidate embeddings: alternatively try passing cls to relation module only
        candidate_embs = self.get_span_emb(transformer_features, candidate_starts, candidate_ends)
        candidate_embs = self.span_relu(self.span_linear_1(candidate_embs))
        # Apply PredictionHead
        mention_labels = self.get_mention_labels(candidate_spans, gold_starts, gold_ends, gold_labels).to(self.device)
        relation_labels = self.get_relation_labels(candidate_starts, candidate_ends, relations)
        
        loss, preds = self.predictor(candidate_starts, candidate_ends, candidate_embs, 
                            transformer_features, cls_embs, tokenindex_to_clsindices,
                            mention_labels, relation_labels, is_training, 
                            self.in_ned_pretraining)

        return loss, [preds[l] for l in OUTPUT_FIELDS]
        
    def predict(self, instance):
        self.eval()
        # Apply the model
        tensorized_example = [b.to(self.device) for b in instance.example]
        tensorized_example.append(instance.all_relations)
        tensorized_example.append(instance)
        tensorized_example.append(False) # is_training
        preds = self.forward(*tensorized_example)[1]
        preds = [x.cpu().data.numpy() for x in preds]
        mention_starts, mention_ends, mention_labels, pair_relation_labels, _, pair_relation_scores = preds
        nb_mentions = len(mention_labels)
      
        loc2label = {}
        for i in range(nb_mentions):
            loc2label[(mention_starts[i], mention_ends[i])] = mention_labels[i]

        # Initialize sample to be returned
        interactions, entities = [], []
        sample = {
            'id': instance.id, 'text': instance.text,
            'interactions': interactions, 'entities': entities
        }

        # Get clusters
        predicted_clusters, mention_to_predicted = [], {}
        for m_start, m_end in zip(mention_starts, mention_ends):
            if not (m_start, m_end) in mention_to_predicted:
                singleton_cluster = (m_start, m_end)
                predicted_clusters.append(singleton_cluster)
                mention_to_predicted[(m_start, m_end)] = singleton_cluster
            else:
                print("repeated mstart, mend")

        # Populate entities
        mention2entityid = {}
        for entityid, cluster in enumerate(predicted_clusters):
            start_token, end_token = cluster
            start_char = instance.tokenization['token2startchar'][start_token]
            end_char = instance.tokenization['token2endchar'][end_token]
            mention2entityid[(start_token, end_token)] = entityid
            entity_name = instance.text[start_char:end_char]
            entity_label = self.configs['entity_types'][loc2label[(start_token, end_token)]]
            if entity_label != NOT_ENTITY:
                entities.append({
                    'label': entity_label,
                    'name': entity_name, 
                    'entity_id': entityid,
                    'start': start_char,
                    'end': end_char
                })

        # Populate interactions
        pred_interactions = {}
        pred_interactionscores={}
        for i in range(nb_mentions):
            # start_idx = i #if self.configs['symmetric_relation'] else 0
            for j in range(0, nb_mentions):
                loci = mention_starts[i], mention_ends[i]
                entityi = mention2entityid[loci]
                locj = mention_starts[j], mention_ends[j]
                entityj = mention2entityid[locj]
                if not (entityi, entityj) in pred_interactions:
                    pred_interactions[(entityi, entityj)] = []
                    pred_interactionscores[(entityi, entityj)] = []
                pred_interactions[(entityi, entityj)].append(self.configs['relation_types'][pair_relation_labels[i, j]])
                pred_interactionscores[(entityi, entityj)].append(pair_relation_scores[i, j, pair_relation_labels[i, j]])
        
        if len(entities) > 0:
            for (a_idx, b_idx) in pred_interactions:
                if len(pred_interactions[(a_idx, b_idx)])>1:
                    print(f"!!!more than one relation predicted {a_idx},{b_idx}: {pred_interactions[(a_idx, b_idx)]}")
                    label,idx = find_majority(pred_interactions[(a_idx, b_idx)])
                    score = pred_interactionscores[(a_idx, b_idx)][idx]

                else:
                    label = pred_interactions[(a_idx, b_idx)][0]
                    score = pred_interactionscores[(a_idx, b_idx)][0]
                if label == NOT_RELATION: continue
                interactions.append({
                    'participants': [a_idx, b_idx],
                    'label': label,
                    'probability':str(round(score,2))
                })
        return sample

    def enumerate_candidate_spans(self, num_tokens, max_span_width, isstartingtoken):
        # Generate candidate spans
        candidate_spans = set([])
        for i in range(num_tokens):
            if isstartingtoken[i]:
                for j in range(i, min(i+max_span_width, num_tokens)):
                    if (j == num_tokens-1) or isstartingtoken[j+1] == 1:
                        candidate_spans.add((i, j))

        return list(candidate_spans)

    def get_mention_labels(self, candidate_spans, gold_starts, gold_ends, gold_labels):
        gold_starts = gold_starts.cpu().data.numpy().tolist()
        gold_ends = gold_ends.cpu().data.numpy().tolist()
        gold_spans = list(zip(gold_starts, gold_ends))
        labels = [0] * len(candidate_spans)
        for idx, (c_start, c_end) in enumerate(candidate_spans):
            if (c_start, c_end) in gold_spans:
                g_index = gold_labels[gold_spans.index((c_start, c_end))]
                labels[idx] = g_index
        labels = torch.LongTensor(labels)
        return labels

    def get_relation_labels(self, candidate_starts, candidate_ends, relations):
        candidate_starts = candidate_starts.cpu().data.numpy().tolist()
        candidate_ends = candidate_ends.cpu().data.numpy().tolist()
        k = len(candidate_starts)
        labels = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                loc1 = candidate_starts[i], candidate_ends[i]
                loc2 = candidate_starts[j], candidate_ends[j]
                if (loc1, loc2) in relations:
                    labels[i,j] = relations[(loc1, loc2)]
                    assert(labels[i,j] != self.configs['notRelationIndex'])

        return torch.LongTensor(labels).to(self.device)

    def get_span_emb(self, context_outputs, span_starts, span_ends):
        span_emb_list = []
        num_tokens = context_outputs.size()[0]
        span_width = span_ends - span_starts + 1

        # Extract the boundary representations for the candidate spans
        span_start_emb = torch.index_select(context_outputs, 0, span_starts)
        span_end_emb = torch.index_select(context_outputs, 0, span_ends)
        assert(span_start_emb.size()[0] == span_end_emb.size()[0])
        span_emb_list.append(span_start_emb)
        span_emb_list.append(span_end_emb)

        # Extract span size feature
        span_width_index = span_width - 1
        span_width_emb = self.span_width_embeddings(span_width_index)
        span_emb_list.append(span_width_emb)

        # Return
        span_emb = torch.cat(span_emb_list, dim=1)
        return span_emb
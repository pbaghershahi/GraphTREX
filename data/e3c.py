import os
import json
import re
from constants import *
from os.path import join
from data.base import DataInstance
from data.helpers import tokenize
import torch,  re

def read_split(raw_insts, tokenizer, configs, split):
  # Construct data_insts
  data_insts = []
  relationsbytype={rtype:0 for rtype in configs['relation_types'] if rtype!=NOT_RELATION}
  avg_entity_len={etype:(0,0) for etype in configs['entity_types'] if etype!=NOT_ENTITY}
  for inst in raw_insts:
    id, text = inst['file_id'], inst['text']    
    raw_entities, raw_timexes, raw_interactions = inst.get('entities', {}),inst.get('timexes', {}), inst.get('relations', {})
    tokenization = tokenize(tokenizer, re.split(" ", text))
    startchar2token, endchar2token = tokenization['startchar2token'], tokenization['endchar2token']

    entities = []
    enumid = 0
    enumid2eid = {}
    eid2enumid = {}

    # Initialize inst_data
    inst_data = {
      'text': text, 'id': id.split(".")[0], 'entities':[], 'interactions': []
    }
    max_len = {}
    for eid, ent in raw_entities.items(): #event, timex, sectimes 
        etype = ent['type']
        # if etype=="BODYPART": etype = "EVENT"
        if etype not in configs["entity_types"]: 
          # if etype not in ['PATIENT', 'H-PROFESSIONAL', 'OTHER']:#ACTOR doesn't participate in temporal relation, BODYPART does
            # print(f"{etype} not in list|eid: {eid}|file: {id}")
          continue
        start, end = int(ent['start']) , int(ent['end'])
        try:
          start_token = startchar2token[start] 
        except:
            print(f"Error mapping token boundary: start:{start} text:{text[start:end]}|eid: {eid}|file: {id}")
            continue
        try:
          end_token = endchar2token[end]
        except:
            print(f"Error mapping token boundary: end:{end} text:{text[start:end]}|eid: {eid}|file: {id}")
            continue
        avg_entity_len[etype]=(avg_entity_len[etype][0]+end_token-start_token+1, avg_entity_len[etype][1]+1)
        if etype not in max_len.keys() or end_token-start_token+1>max_len[etype][0]:
            max_len[etype]= (end_token-start_token+1, text[start:end])
        entities.append({'label': configs["entity_types"].index(etype), 'name':text[start:end],
                          'entity_id': enumid,'start_char': start, 'end_char': end, 
                          'start_token': start_token, 'end_token': end_token})
        inst_data['entities'].append({'label': etype, 'start': start,  'end': end})            
        enumid2eid[enumid] = int(eid)
        eid2enumid[int(eid)] = enumid
        enumid += 1
           
    # Compute relations
    relations = []
    for rid, interaction in raw_interactions.items():
        p1, p2 = int(interaction['head']),int(interaction['tail'])#9460 in EN100017, 11014 in EN100399, 11023 in EN100399 are TIME entities, only present in training set, so excluded
        if p1 in eid2enumid.keys():
          p1 = eid2enumid[p1]
        else: 
          # print(f"p1:{p1} not in entities|{id}|rid:{rid}:{interaction}")
          continue
        if p2 in eid2enumid.keys():
          p2 = eid2enumid[p2]
        else: 
          # print(f"p2:{p2} not in entities|{id}|rid:{rid}:{interaction}")
          continue 
        label = interaction['type'].upper()
        if label == 'BEGINS-ON':
            label = 'BEFORE'
            p1, p2 = p2, p1
        if label == 'ENDS-ON':
            label = 'BEFORE'
        if label in ['CONTAINS', 'SIMULTANEOUS']:
            label = 'OVERLAP'
        relationsbytype[label]+=1
        relations.append({'participants': [p1, p2], 'label': E3C_RELATION_TYPES.index(label)})
        inst_data['interactions'].append({
            'participants': [p1, p2],
            'label': label, 'probability':1
          })
        if configs["FLIP"] and split=="train":
           if label == "OVERLAP":
             relations.append({'participants': [p2, p1], 'label': E3C_RELATION_TYPES.index(label)})
             inst_data['interactions'].append({'participants': [p2, p1], 'label': label, 'probability':1})
    inst_data['interactions'] = [eval(s) for s in set([str(dic) for dic in inst_data['interactions']])]
    data_inst = DataInstance(inst_data, id, text, tokenization, configs["notEntityIndex"], entities, relations, configs["max_tokens"])
    data_insts.append(data_inst)
  return data_insts

def load_e3c_dataset(base_path, tokenizer, split_nb, configs):
  fp = join(base_path, split_nb+".json")
  with open(fp, 'r', encoding='utf-8') as f:
      raw = json.load(f)
  print(f"length of dataset {split_nb}:{len(raw)}")
  return read_split(raw, tokenizer, configs, split_nb)

import os
import json
import re
from constants import *
from os.path import join
from data.base import DataInstance
from data.helpers import tokenize
import torch

def read_split(raw_insts, tokenizer, configs):
  # Construct data_insts
  data_insts = []
  for inst in raw_insts:
    id, text = inst['file_id'], inst['text']
    if configs["transformer"] == "yikuan8/Clinical-Longformer":
        text = text.lower()

    raw_entities, raw_interactions = inst.get('entities', []), inst.get('relations', [])
    tokenization = tokenize(tokenizer, re.split(" ", text)[:configs["MAX_WORDS"]])
    startchar2token, endchar2token = tokenization['startchar2token'], tokenization['endchar2token']

    entities = []
    appeared = {}
    remapped={}
    eid = 0
    num2eid = {}
    eid2num = {}

    # Initialize inst_data
    inst_data = {
      'text': text, 'id': int(id.split(".")[0]), 'entities':[], 'interactions': []
    }
    # discharge_id = ""
    for _, raw_ents in raw_entities.items(): #event, timex, sectimes 
      if len(raw_ents) > 0:
        for ent in raw_ents:
          try:
            start, end = int(ent['start']) , int(ent['end'])
            start_token = startchar2token[start] 
            end_token = endchar2token[end]
            etype = ent['type']
            
            name = ent['text']
            if configs["transformer"] == "yikuan8/Clinical-Longformer":
              name= name.lower()
            if (start, end) in appeared:
              # priority-based assignment
              if appeared[(start, end)]['label']<=TEMPORAL_ENTITY_TYPES.index(etype):
                remapped[ent['id']]=num2eid[appeared[(start, end)]['entity_id']]
                eid2num[ent['id']]=appeared[(start, end)]['entity_id']
                continue
              #else map old id to this
              appeared[(start, end)]['label']=TEMPORAL_ENTITY_TYPES.index(etype)
              remapped[num2eid[appeared[(start, end)]['entity_id']]]= ent['id']
              eid2num[ent['id']]=appeared[(start, end)]['entity_id']
              num2eid[appeared[(start, end)]['entity_id']]=ent['id']
            else:
              appeared[(start, end)] = {
                'label': TEMPORAL_ENTITY_TYPES.index(etype),
                'entity_id': eid, 'name':name,
                'start_char': start, 'end_char': end,
                'start_token': start_token, 'end_token': end_token
                }
              num2eid[eid] = ent['id']
              eid2num[ent['id']] = eid
              eid += 1
          except:
            continue
            
    for entity in appeared.values():
      entities.append(entity)
      inst_data['entities'].append({
        'label': TEMPORAL_ENTITY_TYPES[entity['label']],
        'names': entity['name'],
        'start': entity['start_char'],
        'end': entity['end_char']
        }
      )
    # Compute relations
    relations = []
    if len(raw_interactions) > 0:
      for interaction in raw_interactions:
        try:
          p1, p2 = id.split(".")[0] + "_" + interaction['head'], id.split(".")[0] + "_" + interaction['tail']
          if p1 in remapped:
            p1=remapped[p1]

          p1 = eid2num[p1]
          if p2 in remapped:
            p2=remapped[p2]

          p2 = eid2num[p2]
          label = interaction['type'].upper()
          if label == 'BEGUN_BY':
            label = 'AFTER'
          if label in {'BEFORE_OVERLAP', 'ENDED_BY'}:
            label = 'BEFORE'
          if label in {'DURING', 'SIMULTANEOUS'}:
            label = 'OVERLAP'
          relations.append({'participants': [p1, p2], 'label': TEMPORAL_RELATION_TYPES.index(label)})
          inst_data['interactions'].append({
              'participants': [p1, p2],
              'label': label, 'probability':1
            })
          if configs["FLIP"]:
            if label == "OVERLAP":
              relations.append({'participants': [p2, p1], 'label': TEMPORAL_RELATION_TYPES.index(label)})
              inst_data['interactions'].append({'participants': [p2, p1], 'label': label, 'probability':1})
            if label =="BEFORE":
              relations.append({'participants': [p2, p1], 'label': TEMPORAL_RELATION_TYPES.index("AFTER")})
              inst_data['interactions'].append({'participants': [p2, p1],'label': "AFTER", 'probability':1})
            if label =="AFTER":
              relations.append({'participants': [p2, p1], 'label': TEMPORAL_RELATION_TYPES.index("BEFORE")})
              inst_data['interactions'].append({'participants': [p2, p1],'label': "BEFORE", 'probability':1})
        except Exception as e:
          # print(eid2num)
          # print(f'{e} skipped interaction {id}:{interaction}')
          continue
    
    inst_data['interactions'] = [eval(s) for s in set([str(dic) for dic in inst_data['interactions']])]
    data_inst = DataInstance(inst_data, id, text, tokenization, entities, relations, configs["max_tokens"])
    data_insts.append(data_inst)
    print(f"#entities:{len(entities)} #relations:{len(relations)}")
  return data_insts

def load_temporal_dataset(base_path, tokenizer, split_nb, configs):
  fp = join(base_path, split_nb+".json")
  if configs["transformer"] == "yikuan8/Clinical-Longformer":
      print("longformer")
  with open(fp, 'r', encoding='utf-8') as f:
      raw = json.load(f)
  print(f"length of dataset {split_nb}:{len(raw)}")
  return read_split(raw, tokenizer, configs)

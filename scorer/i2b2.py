import logging
import torch
import numpy as np
import json, os, re
from constants import *
from utils import *
from json2xml import datainstance2file
from tqdm import tqdm

def do_tempeval(predictions, split, modelname):
    output_json_path = os.path.join(f'output/json', f'{split}_{modelname}_predictions.json')
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=True)

    goldpath = f'output/xml/{split}/gold/'
    syspath = f'output/xml/{split}/system{modelname}/'
    create_dir_if_not_exist(syspath)
    for _, preds in predictions.items():
        datainstance2file(preds, goldpath, syspath)
    create_dir_if_not_exist(f'{os.getcwd()}/logs/')
    create_dir_if_not_exist(f'{os.getcwd()}/logs/tempeval/')
    create_dir_if_not_exist(f'{os.getcwd()}/logs/tempeval/{split}/')
    result_file = f'{os.getcwd()}/logs/tempeval/{split}/{modelname}.txt'
    goldpath = f'{os.getcwd()}/output/xml/{split}/gold/'
    syspath = f'{os.getcwd()}/output/xml/{split}/system{modelname}/'
    print(result_file)
    os.system(' '.join(["python tempeval/i2b2Evaluation.py --tempeval", goldpath, syspath]) + ' > ' + result_file)
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            lines = f.readlines()
            res_idx = 3
            m = re.search(r".*\t(.*)\n.*", lines[-res_idx])
            val = m.group(1)
            
    return float(val)

def get_events(data):
    events = set()
    for cluster in data['entities']:
        events.add((cluster['start'], cluster['end'], cluster['label']))
    return events

def get_tlinks(data):
  relations = []
  for interaction in data['interactions']:
    ta, tb = interaction['participants']
    a_obj, b_obj = data['entities'][ta], data['entities'][tb]
    loc_a = '_'.join([str(a_obj['start']), str(a_obj['end'])])
    loc_b = '_'.join([str(b_obj['start']), str(b_obj['end'])])
    rel = {
        'head': loc_a, 'head_type': a_obj['label'],
        'tail': loc_b, 'tail_type': b_obj['label'],
        'type': interaction['label'], 'prob':interaction["probability"]
    }
    relations.append(rel)
  return relations

def evaluate_i2b2(model, dataset, split, modelname, tempeval = False):
    num_docs = len(dataset)
    truth_sentences, pred_sentences, keys = {}, {}, set()
    with torch.no_grad():
        for i in tqdm(range(num_docs)):
            truth_sentences[dataset[i].id] = dataset[i].data
            pred_sentences[dataset[i].id] = model.predict(dataset[i])
            keys.add(dataset[i].id)
    # Evaluate NER
    gt_entities, pred_entities, entity_types = [], [], set()
    for key in keys:
        if key not in pred_sentences:
            print("No prediction for id: {}".format(key))
            continue
        typed_truth = get_events(truth_sentences[key])
        typed_pred = get_events(pred_sentences[key])
         
        batch_gt_entities = []
        for start, end, type in typed_truth:
                if not type == NOT_ENTITY: entity_types.add(type)
                batch_gt_entities.append({
                    'start': start, 'end': end, 'type': type
                })
        gt_entities.append(batch_gt_entities)
        # Update pred_entities
        batch_pred_entities = []
        for start, end, type in typed_pred:
            if not type == NOT_ENTITY: entity_types.add(type)
            batch_pred_entities.append({
                'start': start, 'end': end, 'type': type
            })
        pred_entities.append(batch_pred_entities)
 
    entity_types = list(entity_types)
    entity_score = ner_score(pred_entities, gt_entities, entity_types)

    pred_relations, gt_relations, relation_types = [], [], set()
    for key in keys:
        # Update gt_relations
        typed_truth = get_tlinks(truth_sentences[key])
        gt_relations.append(typed_truth)
        # Update pred_relations
        typed_pred = get_tlinks(pred_sentences[key])
        pred_relations.append(typed_pred)
        # Update relation_types
        for rel in typed_truth:
            if rel['type'] == NOT_RELATION: continue
            relation_types.add(rel['type'])
        for rel in typed_pred:
            if rel['type'] == NOT_RELATION: continue
            relation_types.add(rel['type'])
    relation_types = list(relation_types)
    relation_score = re_score(pred_relations, gt_relations, relation_types)

    # Macro scores
    m_micro = entity_score['ALL']['f1']#['Macro_f1']#everyone uses micro, try using ['f1']
    m_macro = entity_score['ALL']['Macro_f1']
    ent_each_scores = {ent_type:round(entity_score[ent_type]["f1"], 2) for ent_type in entity_types}
    print(f"Mention Macro F1 = {m_macro}")
    print(f"Mention class-wise Micro F1 = {ent_each_scores}")

    r_micro = relation_score['ALL']['f1']#['Macro_f1']#use micro
    rel_each_scores = {rel_type: round(relation_score[rel_type]["f1"],2) for rel_type in relation_types}
    print(f"Relation Macro F1 = {relation_score['ALL']['Macro_f1']}")
    print(f"Relation class-wise Micro F1: {rel_each_scores}")
    print('Mention Score (F1) = {} | Relation Extraction (F1) = {}'.format(round(m_micro,2), round(r_micro,2)))
    rel_score = r_micro
    if tempeval:
        rel_score =  do_tempeval(pred_sentences, split, modelname)
    return pred_sentences, m_micro, rel_score

def ner_score(pred_entities, gt_entities, entity_types):
    """Evaluate NER predictions
    Args:
        pred_entities (list) :  list of list of predicted entities (several entities in each sentence)
        gt_entities (list) :    list of list of ground truth entities
            entity = {"start": start_idx (inclusive),
                      "end": end_idx (exclusive),
                      "type": ent_type}
        entity_types :       Entity Types """
    assert len(pred_entities) == len(gt_entities)

    scores = {ent: {"tp": 0, "fp": 0, "fn": 0} for ent in entity_types + ["ALL"]}

    # Count GT entities and Predicted entities
    n_sents = len(gt_entities)
    n_phrases = sum([len([ent for ent in sent]) for sent in gt_entities])
    n_found = sum([len([ent for ent in sent]) for sent in pred_entities])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_entities, gt_entities):
        for ent_type in entity_types:
            pred_ents = {(ent["start"], ent["end"]) for ent in pred_sent if ent["type"] == ent_type}
            gt_ents = {(ent["start"], ent["end"]) for ent in gt_sent if ent["type"] == ent_type}

            scores[ent_type]["tp"] += len(pred_ents & gt_ents)
            scores[ent_type]["fp"] += len(pred_ents - gt_ents)
            scores[ent_type]["fn"] += len(gt_ents - pred_ents)

    # Compute per entity Precision / Recall / F1
    for ent_type in scores.keys():
        if scores[ent_type]["tp"]:
            scores[ent_type]["p"] = 100 * scores[ent_type]["tp"] / (scores[ent_type]["fp"] + scores[ent_type]["tp"])
            scores[ent_type]["r"] = 100 * scores[ent_type]["tp"] / (scores[ent_type]["fn"] + scores[ent_type]["tp"])
        else:
            scores[ent_type]["p"], scores[ent_type]["r"] = 0, 0

        if not scores[ent_type]["p"] + scores[ent_type]["r"] == 0:
            scores[ent_type]["f1"] = 2 * scores[ent_type]["p"] * scores[ent_type]["r"] / (
                    scores[ent_type]["p"] + scores[ent_type]["r"])
        else:
            scores[ent_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[ent_type]["tp"] for ent_type in entity_types])
    fp = sum([scores[ent_type]["fp"] for ent_type in entity_types])
    fn = sum([scores[ent_type]["fn"] for ent_type in entity_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in entity_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in entity_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in entity_types])

    return scores


def re_score(pred_relations, gt_relations, relation_types, mode="strict"):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}
        relation_types :        Relation Types
        mode (str) :            in 'strict' or 'boundaries' """

    assert mode in ["strict", "boundaries"]

    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                             rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                           rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}
            
            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per entity Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = round(100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"]),2)
            scores[rel_type]["r"] = round(100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"]),2)
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = round(2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"]),2)
        else:
            scores[rel_type]["f1"] = 0
    
    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = round(precision, 2)
    scores["ALL"]["r"] = round(recall,2)
    scores["ALL"]["f1"] = round(f1,2)
    scores["ALL"]["tp"] = round(tp,2)
    scores["ALL"]["fp"] = round(fp,2)
    scores["ALL"]["fn"] = round(fn,2)

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = round(np.mean([scores[ent_type]["f1"] for ent_type in relation_types]),2)
    scores["ALL"]["Macro_p"] = round(np.mean([scores[ent_type]["p"] for ent_type in relation_types]),2)
    scores["ALL"]["Macro_r"] = round(np.mean([scores[ent_type]["r"] for ent_type in relation_types]), 2)

    return scores


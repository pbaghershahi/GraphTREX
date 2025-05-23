from __future__ import print_function

import io
import os
import json
import argparse
import torch
from utils import *
import operator
import time
from constants import *
from transformers import AutoTokenizer
from data import load_data
from models import JointModel
from data import DataInstance, tokenize
from scorer import evaluate
import logging
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_components(model_path, modelname, dataset,  config_name = 'basic'):#use_gold = False,
    print(f"config_name:{config_name} model_path:{model_path}")
    configs = prepare_configs(config_name, dataset, modelname, model_path)#, use_gold = use_gold)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])
    model = JointModel(configs)
    checkpoint = torch.load(os.path.join(configs['save_dir'],configs['modelname']), 
                            map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return tokenizer, model, configs

def evaluate_final(args):
    print(f'Evaluating {args["modelname"]} on {args["split"]} of {args["data"]}')
    
    create_dir_if_not_exist('output')
    create_dir_if_not_exist(f'output/json')
    create_dir_if_not_exist(f'output/json/{args["data"]}')
    if args["data"]in [I2B2]:
        create_dir_if_not_exist('output/xml')
        create_dir_if_not_exist(f'output/xml/{args["data"]}')
        create_dir_if_not_exist(f'output/xml/{args["data"]}/{args["split"]}')
        create_dir_if_not_exist(xmlpath)
        xmlpath = f'output/xml/{args["data"]}/{args["split"]}/system{args["modelname"].replace("model","")}/'
    output_json_path = os.path.join(f'output/json/{args["data"]}', args["split"]+'_'+args["modelname"].replace("model", "")+'_predictions.json')

    tokenizer, model, configs = load_components(args["model_dir"], f'{args["modelname"]}.pt', args["data"], args["version"])#, use_gold = args["use_gold"])
    dataset = load_data(args["split"], tokenizer, configs)
    
    starttime = time.time()
    predictions, _, _ = evaluate(model, dataset, configs, "test")
    endtime = time.time()
    print(f"Total time: {divmod(endtime-starttime, 3600)}")
    # if not args["tempeval"]:
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=True)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_dir', type=str,
                        help='Path to the pretrained models.', required = True)
    parser.add_argument('-s', '--split', default='test', type =str, choices = SPLITS)
    parser.add_argument('-d', '--dataset', default=I2B2, choices=DATASETS)
    parser.add_argument('-n', '--modelname', default='model_basic', type =str)
    parser.add_argument('-v', '--version', default='basic', type =str)
    parser.add_argument('-t', '--tempeval', default='True', type =str)
    # parser.add_argument('-g', '--use_gold', default='False', type =str)
    args = parser.parse_args()
    args = {"split":args.split, "modelname":args.modelname, "data":args.dataset,
            "model_dir": args.model_dir, "version": args.version,# "use_gold": args.use_gold,
            "tempeval":True}
    evaluate_final(args)

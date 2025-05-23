import os
import time
import torch
import random
import math
from utils import *
from constants import *
from transformers import AutoTokenizer
from data import load_data
from scorer import evaluate
from models import JointModel
from argparse import ArgumentParser
from evaluate_model import evaluate_final

# Main Functions
def train(configs):
    torch.manual_seed(configs['seed'])
    np.random.seed(configs['seed'])
    random.seed(configs['seed'])
    torch.cuda.manual_seed(configs['seed'])

    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])
    if configs['dataset'] == ADE:
        train, dev = load_data(configs['split_nb'], tokenizer, configs)
    else: 
        train = load_data('train',  tokenizer, configs)
        dev = load_data('dev', tokenizer, configs)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model = JointModel(configs)
    print('Train Size = {} | Dev Size = {}'.format(len(train), len(dev)))
    print('Initialize a new model | {} parameters'.format(get_n_params(model)))
 
    best_dev_score, best_dev_m_score, best_dev_rel_score = 0, 0, 0
    if configs["use_pretrained"]:
        if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
            checkpoint = torch.load(PRETRAINED_MODEL, map_location=model.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print('Reloaded a pretrained model')
            print('Evaluation on the dev set')
            startevaltime= time.time()
            _, dev_m_score, dev_rel_score = evaluate(model, dev, configs)
            print(f"Evaluation time :{time.time()-startevaltime}")
            best_dev_score = (dev_m_score + dev_rel_score) / 2.0
            print(f'all: {best_dev_score}, mention {best_dev_m_score}, relation {best_dev_rel_score}')
            if configs['freeze_encoder']:
                for name, param in model.named_parameters():
                    # print(f"name:{name} param:{param}")
                    if param.requires_grad:
                        if "transformer" in name:
                            param.requires_grad = False
        # Prepare the optimizer and the scheduler
    num_train_docs = len(train)
    num_epoch_steps = math.ceil(num_train_docs / configs['batch_size'])
    num_train_steps = int(num_epoch_steps * configs['epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = model.get_optimizer(num_warmup_steps, num_train_steps)
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start training
    starttrainingtime = time.time()
    accumulated_loss = RunningAverage()
    iters, batch_loss = 0, 0
    # epoch_losses=[]
    for i in range(configs['epochs']):
        print()
        print('Starting epoch {}'.format(i+1), flush=True)
        model.in_ned_pretraining = i < configs['ned_pretrain_epochs']
        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)

        for train_idx in train_indices:
            iters += 1
            tensorized_example = [b.to(model.device) for b in train[train_idx].example] 
            tensorized_example.append(train[train_idx].all_relations)
            tensorized_example.append(train[train_idx])
            tensorized_example.append(True) # is_training
            
            iter_loss = model(*tensorized_example)[0]
            iter_loss /= configs['batch_size']
            iter_loss.backward()
            # sumgrads = 0
            # for name, param in model.named_parameters():
            #    if param.requires_grad and param.grad is not None:
            #        norm = torch.norm(param.grad)
                #    print(f"name:{name} param:{param} norm:{norm}")
                #    sumgrads += norm
            # print(f"sumgrads:{sumgrads}") 
            batch_loss += iter_loss.data.item()
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
            # Report loss
            if iters % configs['report_frequency'] == 0:
                print('{} Average Loss = {}'.format(iters, accumulated_loss()), flush=True)
                accumulated_loss = RunningAverage()
        # Evaluation after each epoch
        with torch.no_grad():
            _, dev_m_score, dev_rel_score = evaluate(model, dev, configs)
            dev_score = (dev_m_score + dev_rel_score) / 2.0

        # Save model if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_dev_m_score = dev_m_score
            best_dev_rel_score = dev_rel_score
            # Save the model
            save_path = join(configs['save_dir'], '{}.pt'.format(configs['modelname']))
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)
        # epoch_losses.append(accumulated_loss())
        print(f"Total hours elapsed: {divmod(time.time()-starttrainingtime, 3600)}")

    print(f"Training time in seconds: {time.time()-starttrainingtime} ")
    # plot_loss(epoch_losses)
    return {'all': best_dev_score, 'mention': best_dev_m_score, 'relation': best_dev_rel_score}

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='basic')
    parser.add_argument('-d', '--dataset', default=TEMPORAL, choices=DATASETS)
    parser.add_argument('-n', '--modelname', default="model") # Only affect ADE dataset
    parser.add_argument('-m', '--models_dir', default = "tmp")
    # parser.add_argument('-g', '--use_gold', default='False', type =str)
    args = parser.parse_args()
    print(f"dataset:{args.dataset}")
    # Start training
    configs = prepare_configs(args.config_name, args.dataset, 
                              args.modelname, args.models_dir)#, args.use_gold)
    print(configs)
    train(configs)
    
    evaluate_final({"split":"test", "modelname":args.modelname, "data":configs["dataset"], 
                    "model_dir": args.models_dir, "version": args.config_name,# "use_gold":args.use_gold,
                    "tempeval":True})

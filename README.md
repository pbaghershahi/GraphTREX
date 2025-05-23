#jointTIEgnn

step1: modify configs/basic.conf file to select appropriate options

step 2: example command for training a model named graphtrex:
CUDA_VISIBLE_DEVICES=0 python trainer.py -m tmp -n graphtrex>logs/train/graphtrex.txt 2>logs/error.txt

step 3: example command for evaluating the biomedbert model:
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py -m tmp -s test -n graphtrex>logs/test/graphtrex.txt 2>logs/error.txt
[Also creates file with final evaluation results in logs/tempeval/test folder]

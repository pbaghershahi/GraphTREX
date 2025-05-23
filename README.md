#jointTIEgnn

step 1: Preprocess the data and create .json files using resources/<data>/<data>_preprocess.ipynb. Save .json files in the respective resource/<data> folder. For I2b2, also copy the xml files for test split in output/xml/test/gold folder for tempeval script.

step 2: modify configs/i2b2.conf or e3c.conf file to select appropriate options

step 3: example command for training a model named graphtrex with the i2b2 dataset (or select e3c)--the command also performs evaluation at the end on the test set. This command also creates file with final evaluation results in logs/tempeval/test folder:
CUDA_VISIBLE_DEVICES=0 python trainer.py -d i2b2 -m tmp -n graphtrex >logs/train/graphtrex.txt 2>logs/error.txt

step 4: example command for only evaluating the graphtrex model on test set:
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py -d i2b2 -m tmp -s test -n graphtrex>logs/test/graphtrex.txt 2>logs/error.txt


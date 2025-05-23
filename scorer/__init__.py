from constants import *
from scorer.i2b2 import evaluate_i2b2
from scorer.e3c import evaluate_e3c

def evaluate(model, data ,configs, split=""):
    if configs["dataset"] == I2B2:
        return evaluate_i2b2(model, data, split, configs["modelname"], tempeval = configs["tempeval"])
    if configs["dataset"] == E3C:
        return evaluate_e3c(model, data, configs)
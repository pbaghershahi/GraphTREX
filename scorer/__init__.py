from constants import *
from scorer.temporal import evaluate_temporal
from scorer.e3c import evaluate_e3c

def evaluate(model, data ,configs):
    if configs["dataset"] == TEMPORAL:
        return evaluate_temporal(model, data, configs["modelname"], tempeval = configs["tempeval"])
    if configs["dataset"] == E3C:
        return evaluate_e3c(model, data, configs)
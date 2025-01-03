import os
import torch
import numpy as np

from prune_kd import prune_and_knowledge_distillation
from ft import dataset_loading, fine_tuning, model_test_print, model_eval



if __name__ == '__main__':
    train_data, valid_data, test_data = dataset_loading()
    # fine_tuning(train_data)
    # model_test_print(valid_data)
    # model_eval(valid_data)
    prune_and_knowledge_distillation(valid_data, test_data)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multilabel_softmargin_loss_inputs():
    list_of_inputs = []

    input1 = np.random.rand(10, 5).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(10, 5)).astype(np.float32)
    input_dict1 = {
        'weight': np.ones(5).astype(np.float32),
        'size_average': True,
        'reduce': True,
        'reduction': 'mean',
        'input': input1,
        'target': target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(5, 10).astype(np.float32)
    target2 = np.random.randint(0, 2, size=(5, 10)).astype(np.float32)
    input_dict2 = {
        'weight': np.random.rand(10).astype(np.float32),
        'size_average': False,
        'reduce': False,
        'reduction': 'sum',
        'input': input2,
        'target': target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(20, 8).astype(np.float32)
    target3 = np.random.randint(0, 2, size=(20, 8)).astype(np.float32)
    input_dict3 = {
        'weight': np.zeros(8).astype(np.float32),
        'size_average': True,
        'reduce': True,
        'reduction': 'none',
        'input': input3,
        'target': target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 1).astype(np.float32)
    target4 = np.array([[1]]).astype(np.float32)
    input_dict4 = {
        'weight': np.array([1.0]).astype(np.float32),
        'size_average': True,
        'reduce': True,
        'reduction': 'mean',
        'input': input4,
        'target': target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(32, 64).astype(np.float32)
    target5 = np.random.randint(0, 2, size=(32, 64)).astype(np.float32)
    input_dict5 = {
        'weight': np.random.rand(64).astype(np.float32),
        'size_average': False,
        'reduce': True,
        'reduction': 'mean',
        'input': input5,
        'target': target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(16, 32).astype(np.float32)
    target6 = np.random.randint(0, 2, size=(16, 32)).astype(np.float32)
    input_dict6 = {
        'weight': np.ones(32).astype(np.float32),
        'size_average': True,
        'reduce': False,
        'reduction': 'sum',
        'input': input6,
        'target': target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 4).astype(np.float32)
    target7 = np.random.randint(0, 2, size=(4, 4)).astype(np.float32)
    input_dict7 = {
        'weight': np.random.rand(4).astype(np.float32),
        'size_average': False,
        'reduce': False,
        'reduction': 'none',
        'input': input7,
        'target': target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(8, 2).astype(np.float32)
    target8 = np.random.randint(0, 2, size=(8, 2)).astype(np.float32)
    input_dict8 = {
        'weight': np.ones(2).astype(np.float32),
        'size_average': True,
        'reduce': True,
        'reduction': 'mean',
        'input': input8,
        'target': target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(2, 6).astype(np.float32)
    target9 = np.random.randint(0, 2, size=(2, 6)).astype(np.float32)
    input_dict9 = {
        'weight': np.random.rand(6).astype(np.float32),
        'size_average': False,
        'reduce': True,
        'reduction': 'sum',
        'input': input9,
        'target': target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(64, 1).astype(np.float32)
    target10 = np.random.randint(0, 2, size=(64, 1)).astype(np.float32)
    input_dict10 = {
        'weight': np.array([1.0]).astype(np.float32),
        'size_average': True,
        'reduce': True,
        'reduction': 'mean',
        'input': input10,
        'target': target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.MultiLabelSoftMarginLoss"] = multilabel_softmargin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MultiLabelSoftMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MultiLabelSoftMarginLoss'.")


check_valid('torch.nn.MultiLabelSoftMarginLoss', generated_inputs['torch.nn.MultiLabelSoftMarginLoss'], lib="torch", suffix=0)

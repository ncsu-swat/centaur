
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def multilabel_margin_loss_inputs():
    list_of_inputs = []

    input1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target1 = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int64)
    size_average1 = True
    reduce1 = True
    reduction1 = 'mean'
    input_dict1 = {
        "input": input1,
        "target": target1,
        "size_average": size_average1,
        "reduce": reduce1,
        "reduction": reduction1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32)
    target2 = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    size_average2 = False
    reduce2 = False
    reduction2 = 'sum'
    input_dict2 = {
        "input": input2,
        "target": target2,
        "size_average": size_average2,
        "reduce": reduce2,
        "reduction": reduction2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], dtype=np.float32)
    target3 = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.int64)
    size_average3 = True
    reduce3 = True
    reduction3 = 'none'
    input_dict3 = {
        "input": input3,
        "target": target3,
        "size_average": size_average3,
        "reduce": reduce3,
        "reduction": reduction3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    target4 = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int64)
    size_average4 = False
    reduce4 = True
    reduction4 = 'mean'
    input_dict4 = {
        "input": input4,
        "target": target4,
        "size_average": size_average4,
        "reduce": reduce4,
        "reduction": reduction4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.functional.multilabel_margin_loss"] = multilabel_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.multilabel_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.multilabel_margin_loss'.")


check_valid('torch.nn.functional.multilabel_margin_loss', generated_inputs['torch.nn.functional.multilabel_margin_loss'], lib="torch", suffix=0)

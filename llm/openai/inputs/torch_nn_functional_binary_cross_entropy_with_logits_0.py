
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    input1 = np.array([[-1.0, 0.5, 1.2]]).astype(np.float32)
    target1 = np.array([[0.0, 1.0, 0.0]]).astype(np.float32)
    weight1 = np.array([1.0, 2.0, 1.0]).astype(np.float32)
    size_average1 = True
    reduce1 = True
    reduction1 = 'mean'
    pos_weight1 = np.array([1.0, 1.0, 1.0]).astype(np.float32)

    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "size_average": size_average1,
        "reduce": reduce1,
        "reduction": reduction1,
        "pos_weight": pos_weight1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.1, -0.8, 2.5], [0.3, 1.1, -0.2]]).astype(np.float32)
    target2 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).astype(np.float32)
    weight2 = np.array([0.5, 1.0, 0.5]).astype(np.float32)
    size_average2 = False
    reduce2 = False
    reduction2 = 'sum'
    pos_weight2 = np.array([2.0, 1.0, 2.0]).astype(np.float32)

    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "size_average": size_average2,
        "reduce": reduce2,
        "reduction": reduction2,
        "pos_weight": pos_weight2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-2.0, 1.5, -0.7, 0.9]]).astype(np.float32)
    target3 = np.array([[0.0, 1.0, 0.0, 1.0]]).astype(np.float32)
    weight3 = np.array([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    size_average3 = True
    reduce3 = True
    reduction3 = 'none'
    pos_weight3 = np.array([1.0, 2.0, 1.0, 2.0]).astype(np.float32)

    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": weight3,
        "size_average": size_average3,
        "reduce": reduce3,
        "reduction": reduction3,
        "pos_weight": pos_weight3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.0, 0.5, 1.2, -0.3, 2.1]]).astype(np.float32)
    target4 = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]]).astype(np.float32)
    weight4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    size_average4 = False
    reduce4 = True
    reduction4 = 'mean'
    pos_weight4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).astype(np.float32)

    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "size_average": size_average4,
        "reduce": reduce4,
        "reduction": reduction4,
        "pos_weight": pos_weight4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[-0.5], [1.0], [-2.0]]).astype(np.float32)
    target5 = np.array([[0.0], [1.0], [0.0]]).astype(np.float32)
    weight5 = np.array([1.0]).astype(np.float32)
    size_average5 = True
    reduce5 = True
    reduction5 = 'sum'
    pos_weight5 = np.array([2.0]).astype(np.float32)

    input_dict5 = {
        "input": input5,
        "target": target5,
        "weight": weight5,
        "size_average": size_average5,
        "reduce": reduce5,
        "reduction": reduction5,
        "pos_weight": pos_weight5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1.0, 0.5, 1.2]]).astype(np.float32)
    target6 = np.array([[0.0, 1.0, 0.0]]).astype(np.float32)
    weight6 = np.array([1.0, 2.0, 1.0]).astype(np.float32)
    size_average6 = True
    reduce6 = True
    reduction6 = 'mean'
    pos_weight6 = np.array([1.5, 1.0, 1.5]).astype(np.float32)

    input_dict6 = {
        "input": input6,
        "target": target6,
        "weight": weight6,
        "size_average": size_average6,
        "reduce": reduce6,
        "reduction": reduction6,
        "pos_weight": pos_weight6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[0.1, -0.8, 2.5], [0.3, 1.1, -0.2]]).astype(np.float32)
    target7 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).astype(np.float32)
    weight7 = np.array([0.5, 1.0, 0.5]).astype(np.float32)
    size_average7 = False
    reduce7 = False
    reduction7 = 'sum'
    pos_weight7 = np.array([2.5, 1.0, 2.5]).astype(np.float32)

    input_dict7 = {
        "input": input7,
        "target": target7,
        "weight": weight7,
        "size_average": size_average7,
        "reduce": reduce7,
        "reduction": reduction7,
        "pos_weight": pos_weight7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[-2.0, 1.5, -0.7, 0.9]]).astype(np.float32)
    target8 = np.array([[0.0, 1.0, 0.0, 1.0]]).astype(np.float32)
    weight8 = np.array([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    size_average8 = True
    reduce8 = True
    reduction8 = 'none'
    pos_weight8 = np.array([1.5, 2.0, 1.5, 2.0]).astype(np.float32)

    input_dict8 = {
        "input": input8,
        "target": target8,
        "weight": weight8,
        "size_average": size_average8,
        "reduce": reduce8,
        "reduction": reduction8,
        "pos_weight": pos_weight8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[-1.0, 0.5, 1.2, -0.3, 2.1]]).astype(np.float32)
    target9 = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]]).astype(np.float32)
    weight9 = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    size_average9 = False
    reduce9 = True
    reduction9 = 'mean'
    pos_weight9 = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).astype(np.float32)

    input_dict9 = {
        "input": input9,
        "target": target9,
        "weight": weight9,
        "size_average": size_average9,
        "reduce": reduce9,
        "reduction": reduction9,
        "pos_weight": pos_weight9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[-0.5, 2.3, -1.7]]).astype(np.float32)
    target10 = np.array([[0.0, 1.0, 0.0]]).astype(np.float32)
    weight10 = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average10 = True
    reduce10 = False
    reduction10 = 'sum'
    pos_weight10 = np.array([1.0, 2.0, 1.0]).astype(np.float32)

    input_dict10 = {
        "input": input10,
        "target": target10,
        "weight": weight10,
        "size_average": size_average10,
        "reduce": reduce10,
        "reduction": reduction10,
        "pos_weight": pos_weight10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy_with_logits"] = binary_cross_entropy_with_logits_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.binary_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy_with_logits'.")


check_valid('torch.nn.functional.binary_cross_entropy_with_logits', generated_inputs['torch.nn.functional.binary_cross_entropy_with_logits'], lib="torch", suffix=0)

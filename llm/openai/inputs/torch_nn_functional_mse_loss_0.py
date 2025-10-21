
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def mse_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
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

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    target2 = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
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

    input3 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    target3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
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

    input4 = np.random.rand(5, 5).astype(np.float32)
    target4 = np.random.rand(5, 5).astype(np.float32)
    size_average4 = True
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

    input5 = np.array([1.0], dtype=np.float64)
    target5 = np.array([2.0], dtype=np.float64)
    size_average5 = True
    reduce5 = True
    reduction5 = 'mean'

    input_dict5 = {
        "input": input5,
        "target": target5,
        "size_average": size_average5,
        "reduce": reduce5,
        "reduction": reduction5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    target6 = np.array([0.0, -1.0, 2.0], dtype=np.float32)
    size_average6 = False
    reduce6 = True
    reduction6 = 'mean'

    input_dict6 = {
        "input": input6,
        "target": target6,
        "size_average": size_average6,
        "reduce": reduce6,
        "reduction": reduction6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2, 3, 4).astype(np.float32)
    target7 = np.random.rand(2, 3, 4).astype(np.float32)
    size_average7 = True
    reduce7 = True
    reduction7 = 'mean'

    input_dict7 = {
        "input": input7,
        "target": target7,
        "size_average": size_average7,
        "reduce": reduce7,
        "reduction": reduction7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0, 2.0], dtype=np.float16)
    target8 = np.array([1.0, 3.0], dtype=np.float16)
    size_average8 = True
    reduce8 = True
    reduction8 = 'mean'

    input_dict8 = {
        "input": input8,
        "target": target8,
        "size_average": size_average8,
        "reduce": reduce8,
        "reduction": reduction8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    target9 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    size_average9 = True
    reduce9 = True
    reduction9 = 'mean'

    input_dict9 = {
        "input": input9,
        "target": target9,
        "size_average": size_average9,
        "reduce": reduce9,
        "reduction": reduction9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target10 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    size_average10 = False
    reduce10 = False
    reduction10 = 'none'

    input_dict10 = {
        "input": input10,
        "target": target10,
        "size_average": size_average10,
        "reduce": reduce10,
        "reduction": reduction10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.mse_loss"] = mse_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.mse_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.mse_loss'.")


check_valid('torch.nn.functional.mse_loss', generated_inputs['torch.nn.functional.mse_loss'], lib="torch", suffix=0)

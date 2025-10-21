
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pairwise_distance_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    p1 = 2.0
    eps1 = 1e-6
    keepdim1 = True
    
    input_dict1 = {
        "x1": input1,
        "x2": input2,
        "p": p1,
        "eps": eps1,
        "keepdim": keepdim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input4 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    p3 = 1.0
    eps3 = 1e-8
    keepdim3 = False
    
    input_dict3 = {
        "x1": input3,
        "x2": input4,
        "p": p3,
        "eps": eps3,
        "keepdim": keepdim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input5 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input6 = np.array([[5.0, -6.0], [-7.0, 8.0]])
    p5 = 2.0
    eps5 = 1e-5
    keepdim5 = True

    input_dict5 = {
        "x1": input5,
        "x2": input6,
        "p": p5,
        "eps": eps5,
        "keepdim": keepdim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input7 = np.array([[1.0]])
    input8 = np.array([[2.0]])
    p7 = 2.0
    eps7 = 1e-7
    keepdim7 = False

    input_dict7 = {
        "x1": input7,
        "x2": input8,
        "p": p7,
        "eps": eps7,
        "keepdim": keepdim7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input9 = np.array([[1.0, 2.0, 3.0, 4.0]])
    input10 = np.array([[5.0, 6.0, 7.0, 8.0]])
    p9 = 2.0
    eps9 = 1e-4
    keepdim9 = True

    input_dict9 = {
        "x1": input9,
        "x2": input10,
        "p": p9,
        "eps": eps9,
        "keepdim": keepdim9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input11 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    input12 = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    p11 = 3.0
    eps11 = 1e-3
    keepdim11 = False

    input_dict11 = {
        "x1": input11,
        "x2": input12,
        "p": p11,
        "eps": eps11,
        "keepdim": keepdim11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    input13 = np.array([[0.0, 0.0], [0.0, 0.0]])
    input14 = np.array([[0.0, 0.0], [0.0, 0.0]])
    p13 = 2.0
    eps13 = 1e-6
    keepdim13 = True

    input_dict13 = {
        "x1": input13,
        "x2": input14,
        "p": p13,
        "eps": eps13,
        "keepdim": keepdim13
    }
    list_of_inputs.append(copy.deepcopy(input_dict13))

    input15 = np.random.rand(2, 2)
    input16 = np.random.rand(2, 2)
    p15 = 2.0
    eps15 = 1e-6
    keepdim15 = False

    input_dict15 = {
        "x1": input15,
        "x2": input16,
        "p": p15,
        "eps": eps15,
        "keepdim": keepdim15
    }
    list_of_inputs.append(copy.deepcopy(input_dict15))

    input17 = np.array([[1.0, 2.0, 3.0]])
    input18 = np.array([[4.0, 5.0, 6.0]])
    p17 = 1.0
    eps17 = 1e-5
    keepdim17 = True

    input_dict17 = {
        "x1": input17,
        "x2": input18,
        "p": p17,
        "eps": eps17,
        "keepdim": keepdim17
    }
    list_of_inputs.append(copy.deepcopy(input_dict17))
    
    input19 = np.array([[1.5, 2.5], [3.5, 4.5]])
    input20 = np.array([[5.5, 6.5], [7.5, 8.5]])
    p19 = 2.0
    eps19 = 1e-7
    keepdim19 = False

    input_dict19 = {
        "x1": input19,
        "x2": input20,
        "p": p19,
        "eps": eps19,
        "keepdim": keepdim19
    }
    list_of_inputs.append(copy.deepcopy(input_dict19))

    return list_of_inputs

generated_inputs["torch.nn.functional.pairwise_distance"] = pairwise_distance_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pairwise_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pairwise_distance'.")


check_valid('torch.nn.functional.pairwise_distance', generated_inputs['torch.nn.functional.pairwise_distance'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def diff_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 3, 2])
    n1 = 1
    dim1 = -1
    prepend1 = np.array([0])
    append1 = np.array([4])
    out1 = np.array([])

    input_dict1 = {
        "input": input1,
        "n": n1,
        "dim": dim1,
        "prepend": prepend1,
        "append": append1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2, 3], [3, 4, 5]])
    n2 = 1
    dim2 = 0
    prepend2 = np.array([[0, 0, 0]])
    append2 = np.array([[6, 7, 8]])
    out2 = np.array([])

    input_dict2 = {
        "input": input2,
        "n": n2,
        "dim": dim2,
        "prepend": prepend2,
        "append": append2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([1.0, 2.0, 3.0, 4.0])
    n3 = 2
    dim3 = -1
    prepend3 = np.array([0.0])
    append3 = np.array([5.0])
    out3 = np.array([])
    
    input_dict3 = {
        "input": input3,
        "n": n3,
        "dim": dim3,
        "prepend": prepend3,
        "append": append3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1, -2], [-3, -4]])
    n4 = 1
    dim4 = 1
    prepend4 = np.array([[-5], [-6]])
    append4 = np.array([[-7], [-8]])
    out4 = np.array([])

    input_dict4 = {
        "input": input4,
        "n": n4,
        "dim": dim4,
        "prepend": prepend4,
        "append": append4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 2, 3])
    n5 = 1
    dim5 = -1
    prepend5 = None
    append5 = None
    out5 = np.array([])

    input_dict5 = {
        "input": input5,
        "n": n5,
        "dim": dim5,
        "prepend": prepend5,
        "append": append5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(2, 3, 4)
    n6 = 3
    dim6 = 1
    prepend6 = np.random.rand(2, 1, 4)
    append6 = np.random.rand(2, 1, 4)
    out6 = np.array([])

    input_dict6 = {
        "input": input6,
        "n": n6,
        "dim": dim6,
        "prepend": prepend6,
        "append": append6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, 2, 3, 4, 5])
    n7 = 1
    dim7 = 0
    prepend7 = np.array([0])
    append7 = np.array([6])
    out7 = np.array([])

    input_dict7 = {
        "input": input7,
        "n": n7,
        "dim": dim7,
        "prepend": prepend7,
        "append": append7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1, 2], [3, 4]])
    n8 = 2
    dim8 = 1
    prepend8 = np.array([[0, 0], [0, 0]])
    append8 = np.array([[5, 6], [7, 8]])
    out8 = np.array([])
    
    input_dict8 = {
        "input": input8,
        "n": n8,
        "dim": dim8,
        "prepend": prepend8,
        "append": append8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1, 2, 3])
    n9 = 1
    dim9 = -1
    prepend9 = np.array([0, 0])
    append9 = np.array([4, 5])
    out9 = np.array([])

    input_dict9 = {
        "input": input9,
        "n": n9,
        "dim": dim9,
        "prepend": prepend9,
        "append": append9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1, 2, 3], [4, 5, 6]])
    n10 = 1
    dim10 = 0
    prepend10 = np.array([[0, 0, 0]])
    append10 = np.array([[7, 8, 9]])
    out10 = np.array([])

    input_dict10 = {
        "input": input10,
        "n": n10,
        "dim": dim10,
        "prepend": prepend10,
        "append": append10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.diff"] = diff_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diff' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diff'.")


check_valid('torch.diff', generated_inputs['torch.diff'], lib="torch", suffix=0)

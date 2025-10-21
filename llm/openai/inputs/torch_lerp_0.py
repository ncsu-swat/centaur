
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lerp_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0])
    end1 = np.array([4.0, 5.0, 6.0])
    weight1 = 0.5
    out1 = np.empty_like(input1)
    input_dict1 = {"input": input1, "end": end1, "weight": weight1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    end2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    weight2 = 0.25
    out2 = np.empty_like(input2)
    input_dict2 = {"input": input2, "end": end2, "weight": weight2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-1.0, 0.0, 1.0])
    end3 = np.array([2.0, 3.0, 4.0])
    weight3 = 1.0
    out3 = np.empty_like(input3)
    input_dict3 = {"input": input3, "end": end3, "weight": weight3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0])
    end4 = np.array([10.0])
    weight4 = 0.0
    out4 = np.empty_like(input4)
    input_dict4 = {"input": input4, "end": end4, "weight": weight4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0])
    end5 = np.array([4.0, 5.0, 6.0])
    weight5 = -0.5
    out5 = np.empty_like(input5)
    input_dict5 = {"input": input5, "end": end5, "weight": weight5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0])
    end6 = np.array([3.0, 4.0])
    weight6 = 0.75
    out6 = np.empty_like(input6)
    input_dict6 = {"input": input6, "end": end6, "weight": weight6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    end7 = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight7 = 0.5
    out7 = np.empty_like(input7)
    input_dict7 = {"input": input7, "end": end7, "weight": weight7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0, 2.0, 3.0, 4.0])
    end8 = np.array([5.0, 6.0, 7.0, 8.0])
    weight8 = 0.3
    out8 = np.empty_like(input8)
    input_dict8 = {"input": input8, "end": end8, "weight": weight8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0.0, 0.0, 0.0])
    end9 = np.array([1.0, 1.0, 1.0])
    weight9 = 1.0
    out9 = np.empty_like(input9)
    input_dict9 = {"input": input9, "end": end9, "weight": weight9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.0])
    end10 = np.array([2.0])
    weight10 = -1.0
    out10 = np.empty_like(input10)
    input_dict10 = {"input": input10, "end": end10, "weight": weight10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.lerp"] = lerp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lerp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lerp'.")


check_valid('torch.lerp', generated_inputs['torch.lerp'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def xlog1py_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([0.1, 0.2, 0.3])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[0.1, 0.2], [0.3, 0.4]])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([1.0])
    input2 = np.array([0.1])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([-1.0, -2.0, -3.0])
    input2 = np.array([0.1, 0.2, 0.3])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([-0.1, -0.2, -0.3])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([0.0, 1.0, 2.0])
    input2 = np.array([0.0, 1.0, 2.0])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([1e-6, 1e-5, 1e-4])
    input2 = np.array([1e-6, 1e-5, 1e-4])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([100.0, 200.0, 300.0])
    input2 = np.array([0.01, 0.02, 0.03])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input2 = np.array([[0.1, -0.2], [-0.3, 0.4]])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.0, 2.0, 3.0, 4.0])
    input2 = np.array([0.1, 0.2, 0.3, 0.4])
    input_dict = {"x": input1, "y": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py'.")


check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py'], lib="torch", suffix=0)

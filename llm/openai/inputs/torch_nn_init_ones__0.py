
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def init_ones_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3])
    list_of_inputs.append({"tensor": input1})

    input2 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append({"tensor": input2})

    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    list_of_inputs.append({"tensor": input3})

    input4 = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"tensor": input4})

    input5 = np.ones((5,), dtype=np.int64)
    list_of_inputs.append({"tensor": input5})

    input6 = np.random.rand(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"tensor": input6})

    input7 = np.empty((3, 3), dtype=np.float16)
    list_of_inputs.append({"tensor": input7})

    input8 = np.array([-1, -2, -3])
    list_of_inputs.append({"tensor": input8})

    input9 = np.full((2, 4), 3.14, dtype=np.float32)
    list_of_inputs.append({"tensor": input9})

    input10 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    list_of_inputs.append({"tensor": input10})
    
    input11 = np.array([1, 0, 1], dtype=np.bool_)
    list_of_inputs.append({"tensor": input11})

    return list_of_inputs

generated_inputs["torch.nn.init.ones_"] = init_ones_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.ones_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.ones_'.")


check_valid('torch.nn.init.ones_', generated_inputs['torch.nn.init.ones_'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def prod_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3])
    list_of_inputs.append({"input": input1})

    input2 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append({"input": input2, "dim": 1})

    input3 = np.array([1.0, 2.5, 3.7])
    list_of_inputs.append({"input": input3})

    input4 = np.array([[1.1, 2.2], [3.3, 4.4]])
    list_of_inputs.append({"input": input4, "dim": 1})

    input5 = np.array([-1, 2, -3])
    list_of_inputs.append({"input": input5})

    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    list_of_inputs.append({"input": input6, "dim": 1})

    input7 = np.array([1, 2, 3], dtype=np.float64)
    list_of_inputs.append({"input": input7})

    input8 = np.array([[1, 2], [3, 4]], dtype=np.complex128)
    list_of_inputs.append({"input": input8, "dim": 1})

    input9 = np.array([1, 2, 3])
    list_of_inputs.append({"input": input9, "dtype": np.float32})
    
    input10 = np.array([[1, 2, 3], [4, 5, 6]])
    list_of_inputs.append({"input": input10, "dim": 1})

    input11 = np.array([1,2,3])
    list_of_inputs.append({"input": input11})

    return list_of_inputs

generated_inputs["torch.prod_2"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.prod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.prod_2'.")


check_valid('torch.prod', generated_inputs['torch.prod_2'], lib="torch", suffix=2)

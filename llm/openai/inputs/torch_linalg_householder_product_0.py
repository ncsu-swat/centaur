
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def householder_product_inputs():
    list_of_inputs = []

    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    h1 = np.array([-0.31622777, 0.9486833])
    input_dict1 = {"input": input1, "h": h1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    h2 = np.array([-0.5, 0.5, 0.5])
    input_dict2 = {"input": input2, "h": h2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.linalg.householder_product"] = householder_product_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.householder_product' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.householder_product'.")


check_valid('torch.linalg.householder_product', generated_inputs['torch.linalg.householder_product'], lib="torch", suffix=0)

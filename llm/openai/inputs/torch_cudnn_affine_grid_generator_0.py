
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cudnn_affine_grid_generator_inputs():
    list_of_inputs = []
    
    theta1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    N1 = 1
    C1 = 3
    H1 = 32
    W1 = 32
    
    input_dict1 = {
        "theta": theta1,
        "N": N1,
        "C": C1,
        "H": H1,
        "W": W1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.cudnn_affine_grid_generator"] = cudnn_affine_grid_generator_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cudnn_affine_grid_generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cudnn_affine_grid_generator'.")


check_valid('torch.cudnn_affine_grid_generator', generated_inputs['torch.cudnn_affine_grid_generator'], lib="torch", suffix=0)

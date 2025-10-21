
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def masked_scatter_inputs():
    list_of_inputs = []
    
    input = np.array([1, 2, 3, 4, 5])
    mask = np.array([True, False, True, False, True])
    source = np.array([10, 20, 30, 40, 50])
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[1, 2], [3, 4]])
    mask = np.array([[True, False], [False, True]])
    source = np.array([[10, 20], [30, 40]])
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.rand(3, 4, 5)
    mask = np.random.choice([True, False], size=(3, 4, 5), p=[0.5, 0.5])
    source = np.random.rand(3, 4, 5)
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([[-1, 2], [3, -4]])
    mask = np.array([[True, False], [False, True]])
    source = np.array([[10, 20], [30, 40]])
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=bool)
    source = np.ones((2, 2)) * 5
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([1, 2, 3])
    mask = np.array([False, True, False])
    source = np.array([4, 5, 6])
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, True], [False, False]])
    source = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([1, 0, 1, 0])
    mask = np.array([True, False, True, False])
    source = np.array([9, 8, 7, 6])
    
    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randint(0, 10, size=(5, 5))
    mask = np.random.choice([True, False], size=(5, 5), p=[0.3, 0.7])
    source = np.random.randint(10, 20, size=(5, 5))

    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([1, 2, 3, 4])
    mask = np.array([True, True, False, False])
    source = np.array([5, 6, 7, 8])

    input_dict = {
        "input": input,
        "mask": mask,
        "source": source
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.masked_scatter"] = masked_scatter_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.masked_scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.masked_scatter'.")


check_valid('torch.masked_scatter', generated_inputs['torch.masked_scatter'], lib="torch", suffix=0)

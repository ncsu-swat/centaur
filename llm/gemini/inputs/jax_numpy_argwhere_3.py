
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argwhere_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        "a": np.array([1.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float32),
        "size": 5,
        "fill_value": 0.0
    })
    
    # Input 2
    list_of_inputs.append({
        "a": np.array([[0, 1], [2, 0]], dtype=np.int32),
        "size": 3,
        "fill_value": -1.0
    })
    
    # Input 3
    list_of_inputs.append({
        "a": np.array([[[True, False], [False, True]]], dtype=bool),
        "size": 10,
        "fill_value": 0.0
    })
    
    # Input 4
    list_of_inputs.append({
        "a": np.array([-1.5, 0.0, 3.2], dtype=np.float32),
        "size": 2,
        "fill_value": -999.0
    })
    
    # Input 5
    list_of_inputs.append({
        "a": np.zeros((3, 3), dtype=np.int32),
        "size": 4,
        "fill_value": 1.0
    })
    
    # Input 6
    list_of_inputs.append({
        "a": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "size": 8,
        "fill_value": 0.0
    })
    
    # Input 7
    list_of_inputs.append({
        "a": np.ones((5,), dtype=np.float32),
        "size": 6,
        "fill_value": -2.0
    })
    
    # Input 8
    list_of_inputs.append({
        "a": np.array([[100, 200], [0, 0]], dtype=np.int64),
        "size": 1,
        "fill_value": 0.0
    })
    
    # Input 9
    list_of_inputs.append({
        "a": np.array(1, dtype=np.int32),
        "size": 2,
        "fill_value": -1.0
    })
    
    # Input 10
    list_of_inputs.append({
        "a": np.array([[[-1, 0], [1, 2]]], dtype=np.int32),
        "size": 5,
        "fill_value": 0.0
    })
    
    return list_of_inputs

generated_inputs["jax.numpy.argwhere_3"] = argwhere_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argwhere_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argwhere_3'.")


check_valid('jax.numpy.argwhere', generated_inputs['jax.numpy.argwhere_3'], lib="jax", suffix=3)

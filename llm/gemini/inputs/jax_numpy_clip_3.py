
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def clip_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers, positive min and max
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    input_dict = {'arr': arr, 'min': 2, 'max': 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of floats, negative min, positive max
    arr = np.random.uniform(-10.0, 10.0, size=(3, 3)).astype(np.float32)
    input_dict = {'arr': arr, 'min': -2, 'max': 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array of integers, negative min and max
    arr = np.random.randint(-20, 0, size=(2, 2, 2)).astype(np.int64)
    input_dict = {'arr': arr, 'min': -15, 'max': -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array of floats, min > max
    arr = np.random.randn(2, 2, 2, 2).astype(np.float64)
    input_dict = {'arr': arr, 'min': 5, 'max': 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array, uint8, min/max within uint8 range
    arr = np.array([10, 50, 100, 150, 200, 250], dtype=np.uint8)
    input_dict = {'arr': arr, 'min': 40, 'max': 180}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D array (scalar array)
    arr = np.array(42, dtype=np.int32)
    input_dict = {'arr': arr, 'min': 10, 'max': 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, min/max are equal
    arr = np.random.randint(-5, 5, size=(4, 4)).astype(np.int32)
    input_dict = {'arr': arr, 'min': 0, 'max': 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High dimensional 5D array
    arr = np.random.uniform(-100, 100, size=(2, 2, 2, 2, 2)).astype(np.float32)
    input_dict = {'arr': arr, 'min': -50, 'max': 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, float32, large integer limits
    arr = np.array([-1e5, 0.0, 1e5], dtype=np.float32)
    input_dict = {'arr': arr, 'min': -1000, 'max': 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, int16, min and max as negative and positive values
    arr = np.random.randint(-300, 300, size=(3, 2, 4)).astype(np.int16)
    input_dict = {'arr': arr, 'min': -100, 'max': 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.clip_3"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.clip_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.clip_3'.")


check_valid('jax.numpy.clip', generated_inputs['jax.numpy.clip_3'], lib="jax", suffix=3)

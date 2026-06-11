
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def clamp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, clamp between 0.0 and 1.0
    x = np.random.uniform(-2.0, 2.0, size=(10,)).astype(np.float32)
    input_dict = {
        "min": 0.0,
        "x": x,
        "max": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, clamp between -5.0 and 5.0
    x = np.random.uniform(-10.0, 10.0, size=(4, 4)).astype(np.float32)
    input_dict = {
        "min": -5.0,
        "x": x,
        "max": 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, clamp between -0.5 and 0.5
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 2)).astype(np.float64)
    input_dict = {
        "min": -0.5,
        "x": x,
        "max": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, clamp between 10.0 and 20.0
    x = np.random.uniform(5.0, 25.0, size=(1, 5)).astype(np.float32)
    input_dict = {
        "min": 10.0,
        "x": x,
        "max": 20.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, clamp between -100.0 and 100.0
    x = np.random.uniform(-150.0, 150.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {
        "min": -100.0,
        "x": x,
        "max": 100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D (scalar-like) float32 array, clamp between -1.0 and 1.0
    x = np.array(1.5, dtype=np.float32)
    input_dict = {
        "min": -1.0,
        "x": x,
        "max": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array, clamp between 0.0 and 100.0
    x = np.random.uniform(-10.0, 110.0, size=(50,)).astype(np.float32)
    input_dict = {
        "min": 0.0,
        "x": x,
        "max": 100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array, clamp between -2.5 and 2.5
    x = np.random.uniform(-5.0, 5.0, size=(2, 4, 3)).astype(np.float64)
    input_dict = {
        "min": -2.5,
        "x": x,
        "max": 2.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, clamp between -0.01 and 0.01
    x = np.random.uniform(-0.05, 0.05, size=(8, 8)).astype(np.float32)
    input_dict = {
        "min": -0.01,
        "x": x,
        "max": 0.01
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array with singleton dimension, clamp between 1.5 and 9.5
    x = np.random.uniform(0.0, 11.0, size=(3, 1, 3)).astype(np.float32)
    input_dict = {
        "min": 1.5,
        "x": x,
        "max": 9.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.clamp_2"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clamp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clamp_2'.")


check_valid('jax.lax.clamp', generated_inputs['jax.lax.clamp_2'], lib="jax", suffix=2)

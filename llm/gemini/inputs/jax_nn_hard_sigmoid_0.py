
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hard_sigmoid_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array covering negative, zero, positive regions
    x = np.array([-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array (float64)
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array (float32)
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array (float16)
    x = np.random.randn(2, 4, 8, 8).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Large negative values (saturated at 0)
    x = np.array([-100.0, -50.0, -10.0, -3.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Large positive values (saturated at 1)
    x = np.array([3.1, 10.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Integer array (promoted by JAX to float)
    x = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array with zero dimension
    x = np.empty((3, 0), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array
    x = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.hard_sigmoid"] = hard_sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.hard_sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.hard_sigmoid'.")


check_valid('jax.nn.hard_sigmoid', generated_inputs['jax.nn.hard_sigmoid'], lib="jax", suffix=0)

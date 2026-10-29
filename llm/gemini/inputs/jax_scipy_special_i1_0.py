
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def i1_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive floats
    x = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with negative floats
    x = np.array([-5.0, -2.0, -1.0, -0.5, -0.1], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array with mixed positive and negative values
    x = np.array([[-3.0, -1.0, 0.0], [1.0, 2.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D random array float32
    x = np.random.uniform(-10.0, 10.0, size=(2, 4, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D random array float64
    x = np.random.normal(loc=0.0, scale=2.0, size=(2, 2, 3, 4)).astype(
        np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Integer array (which converts cleanly to real-valued)
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Large positive values float64
    x = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with small fractional values close to zero
    x = np.array([-1e-4, -1e-7, 0.0, 1e-7, 1e-4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.i1"] = i1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i1'.")


check_valid('jax.scipy.special.i1', generated_inputs['jax.scipy.special.i1'], lib="jax", suffix=0)

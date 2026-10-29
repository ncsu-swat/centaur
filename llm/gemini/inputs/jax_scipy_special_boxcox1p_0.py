
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def boxcox1p_inputs():
    list_of_inputs = []

    # Input 1: 1D positive float32 arrays
    x = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    l = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 2: 2D float64 arrays, l = 0 (log1p case)
    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    l = np.zeros((2, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 3: Negative x values strictly within domain (x > -1)
    x = np.array([-0.9, -0.5, -0.1, 0.0], dtype=np.float32)
    l = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 4: Negative l parameter values
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    l = np.array([-2.0, -1.0, -0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 5: 0D scalar arrays
    x = np.array(0.75, dtype=np.float32)
    l = np.array(1.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 6: 3D arrays
    x = np.array(
        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32
    )
    l = np.array(
        [[[1.0, 1.0], [2.0, 2.0]], [[0.5, 0.5], [1.5, 1.5]]], dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 7: Broadcasting (1, 4) with (3, 1)
    x = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    l = np.array([[0.5], [1.0], [1.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 8: Broadcasting scalar array with 2D array
    x = np.array(1.5, dtype=np.float32)
    l = np.array([[-1.0, 0.0], [0.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 9: Small values near zero
    x = np.array([1e-6, 1e-8, 1e-12], dtype=np.float64)
    l = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    # Input 10: 4D arrays
    x = np.full((2, 2, 2, 2), 0.5, dtype=np.float32)
    l = np.full((2, 2, 2, 2), 1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "l": l}))

    return list_of_inputs


generated_inputs["jax.scipy.special.boxcox1p"] = boxcox1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.boxcox1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.boxcox1p'.")


check_valid('jax.scipy.special.boxcox1p', generated_inputs['jax.scipy.special.boxcox1p'], lib="jax", suffix=0)

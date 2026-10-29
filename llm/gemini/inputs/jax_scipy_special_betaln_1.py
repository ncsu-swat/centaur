
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def betaln_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    a = np.array(1.5, dtype=np.float32)
    b = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 2: 1D arrays of the same shape, float32
    a = np.array([1.0, 2.0, 3.5, 0.5], dtype=np.float32)
    b = np.array([0.5, 1.5, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 3: 1D arrays, float64
    a = np.array([10.0, 20.0, 0.1], dtype=np.float64)
    b = np.array([5.0, 15.0, 0.2], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 4: 2D arrays, float32
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 5: 2D broadcasting with 1D
    a = np.ones((3, 4), dtype=np.float32) * 2.0
    b = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 6: Broadcasting (3, 1) and (1, 4)
    a = np.array([[0.5], [1.5], [2.5]], dtype=np.float64)
    b = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 7: 3D arrays
    a = np.full((2, 3, 4), 1.2, dtype=np.float32)
    b = np.full((2, 3, 4), 3.4, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 8: Very small positive values
    a = np.array([1e-4, 1e-3, 1e-2], dtype=np.float32)
    b = np.array([1e-4, 1e-3, 1e-2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 9: Large positive values
    a = np.array([100.0, 200.0, 500.0], dtype=np.float64)
    b = np.array([50.0, 150.0, 250.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 10: 4D array with 1D broadcasting
    a = np.ones((2, 2, 2, 2), dtype=np.float32) * 0.75
    b = np.array([1.25], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    return list_of_inputs


generated_inputs["jax.scipy.special.betaln_1"] = betaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.betaln_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.betaln_1'.")


check_valid('jax.scipy.special.betaln', generated_inputs['jax.scipy.special.betaln_1'], lib="jax", suffix=1)

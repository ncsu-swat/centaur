
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def beta_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays (float32)
    a = np.array(0.5, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 2: 1D arrays (float32)
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 3: 1D arrays with broadcasting (1,) and (4,)
    a = np.array([2.5], dtype=np.float64)
    b = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 4: 2D arrays of identical shape (float64)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 5: 2D arrays with broadcasting (3, 1) and (1, 4)
    a = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    b = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 6: Integer inputs
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 7: Small positive values (float32)
    a = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 8: 3D arrays (float32)
    a = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    b = np.ones((2, 3, 4), dtype=np.float32) * 2.5
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 9: Negative non-integer values
    a = np.array([-0.5, -1.5, -2.5], dtype=np.float64)
    b = np.array([1.2, 2.2, 3.2], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 10: 4D arrays (float32)
    a = np.full((2, 2, 1, 3), 0.75, dtype=np.float32)
    b = np.full((2, 1, 4, 3), 1.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    return list_of_inputs


generated_inputs["jax.scipy.special.beta"] = beta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.beta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.beta'.")


check_valid('jax.scipy.special.beta', generated_inputs['jax.scipy.special.beta'], lib="jax", suffix=0)

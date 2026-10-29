
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hyp1f1_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    a = np.array(1.0, dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 2: 1D arrays (float32)
    a = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 3: 2D arrays (float64)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 4: Broadcasting shapes (1, 3) and (3, 1) and (3, 3)
    a = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)
    b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    x = np.full((3, 3), 0.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 5: 3D arrays
    a = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    b = np.ones((2, 3, 4), dtype=np.float32) * 2.5
    x = np.ones((2, 3, 4), dtype=np.float32) * 0.4
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 6: Convention a = 0, b = 0
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 0.0], dtype=np.float32)
    x = np.array([0.5, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 7: Broadcasting scalar to 2D
    a = np.array(2.0, dtype=np.float64)
    b = np.array(5.0, dtype=np.float64)
    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 8: 4D arrays
    a = np.full((2, 2, 2, 2), 1.0, dtype=np.float32)
    b = np.full((2, 2, 2, 2), 3.0, dtype=np.float32)
    x = np.full((2, 2, 2, 2), 0.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 9: Small fractional values
    a = np.array([0.01, 0.05, 0.1], dtype=np.float64)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    x = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 10: 1D with integer-like float values
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.hyp1f1"] = hyp1f1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.hyp1f1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.hyp1f1'.")


check_valid('jax.scipy.special.hyp1f1', generated_inputs['jax.scipy.special.hyp1f1'], lib="jax", suffix=0)

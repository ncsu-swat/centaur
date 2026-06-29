
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def betainc_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays of float32
    a = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    b = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    x = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32
    a = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    b = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    x = np.random.uniform(0.0, 1.0, size=(3, 3)).astype(np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 arrays
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalars in tensor wrapper)
    a = np.array(2.0, dtype=np.float32)
    b = np.array(3.0, dtype=np.float32)
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting - a is column, b is row, x is 2D
    a = np.array([[0.5], [1.5], [2.5]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting - x is a single element array
    a = np.random.uniform(1.0, 5.0, size=(4, 2)).astype(np.float32)
    b = np.random.uniform(1.0, 5.0, size=(4, 2)).astype(np.float32)
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small parameters a and b
    a = np.random.uniform(0.01, 0.1, size=(5,)).astype(np.float32)
    b = np.random.uniform(0.01, 0.1, size=(5,)).astype(np.float32)
    x = np.random.uniform(0.1, 0.9, size=(5,)).astype(np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large parameters a and b
    a = np.random.uniform(50.0, 100.0, size=(2, 2)).astype(np.float64)
    b = np.random.uniform(50.0, 100.0, size=(2, 2)).astype(np.float64)
    x = np.random.uniform(0.1, 0.9, size=(2, 2)).astype(np.float64)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Edge cases of x (0.0 and 1.0)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-dimensional array (3D)
    a = np.ones((2, 3, 4), dtype=np.float32) * 3.0
    b = np.ones((2, 3, 4), dtype=np.float32) * 4.0
    x = np.random.uniform(0.1, 0.9, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.betainc"] = betainc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.betainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.betainc'.")


check_valid('jax.scipy.special.betainc', generated_inputs['jax.scipy.special.betainc'], lib="jax", suffix=0)

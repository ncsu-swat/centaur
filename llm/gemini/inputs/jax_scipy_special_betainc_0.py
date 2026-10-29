
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def betainc_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    a = np.array(0.5, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    x = np.array(0.3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 2: 1D arrays, float32
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 3: 2D arrays, float64
    a = np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float64)
    b = np.array([[1.5, 2.5], [3.5, 0.5]], dtype=np.float64)
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 4: Broadcasting (3, 1) and (1, 4)
    a = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    b = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    x = np.full((3, 4), 0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 5: Boundary values of x (0 and 1)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 6: 3D arrays, float32
    a = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    b = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    x = np.random.uniform(0.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 7: Large parameters
    a = np.array([10.0, 50.0, 100.0], dtype=np.float64)
    b = np.array([15.0, 40.0, 80.0], dtype=np.float64)
    x = np.array([0.3, 0.6, 0.9], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 8: Small parameter values (near zero)
    a = np.array([1e-3, 1e-2, 0.1], dtype=np.float32)
    b = np.array([1e-2, 1e-1, 0.2], dtype=np.float32)
    x = np.array([0.05, 0.15, 0.25], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 9: Scalar broadcasting with 2D tensor
    a = np.array(2.0, dtype=np.float64)
    b = np.array(5.0, dtype=np.float64)
    x = np.linspace(0.1, 0.9, 6, dtype=np.float64).reshape((2, 3))
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

    # Input 10: 4D tensors, float32
    a = np.random.uniform(0.5, 3.0, size=(2, 1, 3, 2)).astype(np.float32)
    b = np.random.uniform(0.5, 3.0, size=(2, 1, 3, 2)).astype(np.float32)
    x = np.random.uniform(0.1, 0.9, size=(2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "x": x}))

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

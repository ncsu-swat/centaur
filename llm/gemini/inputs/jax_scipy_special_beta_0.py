
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_special_beta_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays of float32, simple positive values
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays of float32
    a = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D arrays of float64
    a = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float64)
    b = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float64)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalars as tensors)
    a = np.array(2.5, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small values close to 0
    a = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    b = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values
    a = np.array([10.0, 20.0, 50.0], dtype=np.float64)
    b = np.array([10.0, 20.0, 50.0], dtype=np.float64)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting (5, 1) and (1, 5)
    a = np.linspace(0.5, 5.0, 5, dtype=np.float32).reshape(5, 1)
    b = np.linspace(0.5, 5.0, 5, dtype=np.float32).reshape(1, 5)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting (3, 4, 5) and (5,)
    a = np.random.uniform(1.0, 5.0, size=(3, 4, 5)).astype(np.float32)
    b = np.random.uniform(1.0, 5.0, size=(5,)).astype(np.float32)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer values represented as float64
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays of float32
    a = np.random.uniform(0.5, 2.5, size=(2, 2, 2, 2)).astype(np.float32)
    b = np.random.uniform(0.5, 2.5, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.beta"] = jax_scipy_special_beta_inputs()

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

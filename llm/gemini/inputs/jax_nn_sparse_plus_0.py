
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sparse_plus_inputs():
    list_of_inputs = []

    # Input 1: 1D array covering all regions: <= -1, between -1 and 1, >= 1
    x = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar array
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array with float64
    x = np.array([[-2.0, -0.5], [0.5, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D random normal array with float32
    x = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array with float16
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Values strictly less than -1
    x = np.array([-10.0, -5.0, -2.0, -1.01], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Values strictly greater than 1
    x = np.array([1.01, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Values strictly in (-1, 1)
    x = np.linspace(-0.99, 0.99, 10, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Array with special values (inf, -inf, nan)
    x = np.array([-np.inf, -2.0, 0.0, 2.0, np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.sparse_plus"] = sparse_plus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.sparse_plus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.sparse_plus'.")


check_valid('jax.nn.sparse_plus', generated_inputs['jax.nn.sparse_plus'], lib="jax", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expit_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative and positive values
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array
    x = np.array([[-2.5, 0.5], [1.2, -3.4]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar (0D array) float32
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float32 array with random values
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: Large positive and negative values (edge cases)
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D float16 array
    x = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Integer array (int32)
    x = np.array([-10, -1, 0, 1, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Integer array (int64) 2D
    x = np.array([[-5, 0], [5, 10]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array with zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with special values like inf and -inf
    x = np.array([-np.inf, 0.0, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.expit"] = expit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.expit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.expit'.")


check_valid('jax.scipy.special.expit', generated_inputs['jax.scipy.special.expit'], lib="jax", suffix=0)

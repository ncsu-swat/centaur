
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def swish_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values (standard range)
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar array (float32)
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of float64 values (including negative values)
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of float32 values
    x = np.random.randn(2, 3, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array (typical image batch format BCHW)
    x = np.random.randn(2, 8, 16, 16).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Float16 dtype
    x = np.array([[-10.0, -5.0], [5.0, 10.0]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Large positive and negative values (saturation regions)
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: All zeros of various shapes
    x = np.zeros((4, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Complex64 values
    x = np.array([-2.0 + 1.0j, 0.0 + 0.0j, 1.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D array
    x = np.random.uniform(-3.0, 3.0, size=(2, 2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array (0 elements)
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.swish"] = swish_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.swish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.swish'.")


check_valid('jax.nn.swish', generated_inputs['jax.nn.swish'], lib="jax", suffix=0)

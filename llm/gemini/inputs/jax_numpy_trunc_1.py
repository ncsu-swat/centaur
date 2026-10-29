
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trunc_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive and negative floats
    x = np.array([-1.7, -1.2, -0.2, 0.0, 0.2, 1.2, 1.7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of float64
    x = np.array([[-2.9, 3.5, 0.0], [4.1, -5.8, 6.7]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (float32)
    x = np.array(-3.85, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array with float16
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D integer array
    x = np.array([-10, -3, 0, 5, 12], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Array with special values (inf, -inf, nan)
    x = np.array(
        [-np.inf, -2.5, 0.0, 2.5, np.inf, np.nan], dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D tensor with float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32) * 10
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D integer array (int64)
    x = np.array([[100, -200], [-300, 400]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array with values close to zero (positive and negative subnormals / small values)
    x = np.array([-1e-7, -0.999, 0.999, 1e-7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: High dimensional 5D array
    x = np.random.uniform(-5.0, 5.0, size=(1, 2, 2, 2, 1)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.trunc_1"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trunc_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trunc_1'.")


check_valid('jax.numpy.trunc', generated_inputs['jax.numpy.trunc_1'], lib="jax", suffix=1)

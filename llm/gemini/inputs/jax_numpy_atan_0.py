
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atan_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive, negative, and zero values
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of float64
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of float16
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array of float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Integer input (int32)
    x = np.array([[-5, -1, 0], [1, 5, 10]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Integer input (int64)
    x = np.array([0, -100, 100, 1000], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex input (complex64)
    x = np.array([1.0 + 2.0j, -0.5 - 1.5j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Complex input (complex128)
    x = np.array([[2.0 - 1.0j, -3.0 + 4.0j], [0.5 + 0.5j, -1.0 - 1.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array with special values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array (float32)
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.atan"] = atan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atan'.")


check_valid('jax.numpy.atan', generated_inputs['jax.numpy.atan'], lib="jax", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arctan_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 with positive, negative, and zero values
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 0D scalar array (float64)
    x = np.array(0.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of float32
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D array with special values (inf, -inf, nan)
    x = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array with integer types (promotes to inexact dtype)
    x = np.arange(-12, 12, dtype=np.int32).reshape(2, 3, 4)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array with complex numbers (complex64)
    x = np.array([2.0 + 7.0j, -1.0 - 1.0j, 0.0 + 0.0j, 3.0 - 2.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D tensor with float16
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 3)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Empty array
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D array of complex128
    x = np.array([[1.0 + 2.0j, -2.0 + 0.5j], [0.0 - 1.0j, 4.0 + 3.0j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array of extreme values (very large and very small)
    x = np.array([1e-15, -1e-15, 1e15, -1e15], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D array of booleans
    x = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.arctan_1"] = arctan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctan_1'.")


check_valid('jax.numpy.arctan', generated_inputs['jax.numpy.arctan_1'], lib="jax", suffix=1)

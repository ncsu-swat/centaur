
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sinh_inputs():
    list_of_inputs = []

    # Input 1: Float32 1D array with positive, negative, and zero values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Float64 2D array with random values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Complex64 1D array
    x = np.array([3.0 - 2.0j, -1.0 + 1.0j, 0.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Int32 1D array
    x = np.array([-5, 0, 5], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Float32 0D array (scalar representation)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Float32 3D array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Complex128 2D array
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Float16 1D array
    x = np.array([-1.5, 0.5, 2.5], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Int64 2D array
    x = np.array([[-3, 2], [1, -4]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Float32 4D array
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.sinh_1"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sinh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sinh_1'.")


check_valid('jax.numpy.sinh', generated_inputs['jax.numpy.sinh_1'], lib="jax", suffix=1)

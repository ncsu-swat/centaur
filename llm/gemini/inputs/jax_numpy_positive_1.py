
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def positive_inputs():
    list_of_inputs = []

    # Input 1: Float32 array, 1D, mixed positive and negative values
    x = np.array([-5.5, 4.0, 7.2, -9.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Float64 array, 2D random
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Int32 array, 3D, mixed values
    x = np.array([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Int64 array, 1D
    x = np.array([100000000, -200000000, 300000000], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Complex64 array, 1D
    x = np.array([1-2j, -3+4j, 5-6j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Complex128 array, 2D
    x = np.array([[1+1j, -2-2j], [3-3j, -4+4j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Float16 array, 4D
    x = np.random.randn(2, 2, 2, 2).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Int16 array, 2D
    x = np.array([[-10, 20], [-30, 40]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Float32 array, 0D (scalar)
    x = np.array(-3.14, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Int8 array, 1D
    x = np.array([-128, 0, 127], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Float32 array, 3D
    x = np.random.randn(2, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.positive_1"] = positive_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.positive_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.positive_1'.")


check_valid('jax.numpy.positive', generated_inputs['jax.numpy.positive_1'], lib="jax", suffix=1)

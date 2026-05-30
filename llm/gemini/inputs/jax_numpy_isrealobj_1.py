
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-1.5, 0.0, 2.3], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D int32 array
    x = np.array([[1, -2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D complex64 array
    x = np.array([[[1+2j, 2], [3, 4]]], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D float64 array (scalar-like)
    x = np.array(3.14159, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex128 array
    x = np.array([0j, 1+0j], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D boolean array
    x = np.ones((2, 2, 2, 2), dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D float16 array
    x = np.random.randn(5, 5).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 3D uint8 array
    x = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D int64 array with negative values
    x = np.array([-10, -20, -30, 40], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D float32 array
    x = np.zeros((1, 2, 1, 3, 1), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.isrealobj_1"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_1'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_1'], lib="jax", suffix=1)

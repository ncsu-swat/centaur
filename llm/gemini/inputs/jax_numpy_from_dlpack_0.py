
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def from_dlpack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, copy=True
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, copy=False
    input_dict = {
        "x": np.random.randn(3, 3).astype(np.float64),
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array, copy=True
    input_dict = {
        "x": np.arange(24).reshape(2, 3, 4).astype(np.int32),
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D float32 array, copy=False
    input_dict = {
        "x": np.array(42.0, dtype=np.float32),
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float16 array, copy=True
    input_dict = {
        "x": np.random.uniform(-1, 1, (2, 2, 2, 2)).astype(np.float16),
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array with negative values, copy=False
    input_dict = {
        "x": np.array([-10, -5, 0, 5, 10], dtype=np.int64),
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D uint8 array, copy=True
    input_dict = {
        "x": np.ones((5, 5), dtype=np.uint8) * 255,
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array with negative values, copy=False
    input_dict = {
        "x": np.random.uniform(-100, 100, (3, 3, 3)).astype(np.float32),
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int8 array, copy=True
    input_dict = {
        "x": np.array([-128, 0, 127], dtype=np.int8),
        "copy": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, copy=False
    input_dict = {
        "x": np.zeros((2, 2, 2, 2, 2), dtype=np.float32),
        "copy": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.from_dlpack"] = from_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.from_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.from_dlpack'.")


check_valid('jax.numpy.from_dlpack', generated_inputs['jax.numpy.from_dlpack'], lib="jax", suffix=0)

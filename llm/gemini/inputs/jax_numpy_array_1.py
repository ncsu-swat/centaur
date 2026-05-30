
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    object_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "object": object_1,
        "dtype": np.float32,
        "copy": True,
        "order": "K",
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, minimum 2 dimensions
    object_2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "object": object_2,
        "dtype": np.int32,
        "copy": False,
        "order": "K",
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array
    object_3 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "object": object_3,
        "dtype": np.float64,
        "copy": True,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D int64 array, expanded to 3D
    object_4 = np.array(42, dtype=np.int64)
    input_dict = {
        "object": object_4,
        "dtype": np.int64,
        "copy": True,
        "order": "K",
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D complex64 array
    object_5 = np.random.randn(2, 2, 2, 2).astype(np.complex64)
    input_dict = {
        "object": object_5,
        "dtype": np.complex64,
        "copy": False,
        "order": "K",
        "ndmin": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D boolean array
    object_6 = np.array([True, False, True], dtype=np.bool_)
    input_dict = {
        "object": object_6,
        "dtype": np.bool_,
        "copy": True,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D uint8 array
    object_7 = np.random.randint(0, 255, size=(3, 3), dtype=np.uint8)
    input_dict = {
        "object": object_7,
        "dtype": np.uint8,
        "copy": True,
        "order": "K",
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float16 array
    object_8 = np.random.randn(2, 3, 2).astype(np.float16)
    input_dict = {
        "object": object_8,
        "dtype": np.float16,
        "copy": False,
        "order": "K",
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int16 array with negative values, expanded to 2D
    object_9 = np.array([-10, 0, 10], dtype=np.int16)
    input_dict = {
        "object": object_9,
        "dtype": np.int16,
        "copy": True,
        "order": "K",
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array
    object_10 = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)
    input_dict = {
        "object": object_10,
        "dtype": np.float32,
        "copy": True,
        "order": "K",
        "ndmin": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.array_1"] = array_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_1'.")


check_valid('jax.numpy.array', generated_inputs['jax.numpy.array_1'], lib="jax", suffix=1)

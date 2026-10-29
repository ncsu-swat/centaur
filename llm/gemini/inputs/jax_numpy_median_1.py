
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def median_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    a = np.array([1.0, 3.0, -2.0, 5.0, 0.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, negative axis, keepdims True
    a = np.array([-10, 20, -30, 40], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": -1,
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 1
    a = np.array(
        [[2.5, -1.2, 4.8], [-3.1, 0.0, 1.1], [5.5, -6.6, 7.7]],
        dtype=np.float64,
    )
    input_dict = {
        "a": a,
        "axis": 1,
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, axis 0, keepdims True
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    input_dict = {
        "a": a,
        "axis": 0,
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, axis 2
    a = np.random.randint(-50, 50, size=(2, 3, 4)).astype(np.int64)
    input_dict = {
        "a": a,
        "axis": 2,
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, negative axis -2, keepdims True
    a = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float64 array, axis 3
    a = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 5)).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 3,
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int16 array with even length along axis 1, keepdims True
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int16)
    input_dict = {
        "a": a,
        "axis": 1,
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array, single element
    a = np.array([42.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, axis 0, keepdims True
    a = np.zeros((4, 4, 4), dtype=np.float32)
    a[0, :, :] = -1.0
    a[1, :, :] = 2.0
    input_dict = {
        "a": a,
        "axis": 0,
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D uint8 array, axis 1
    a = np.random.randint(0, 255, size=(3, 4, 2, 2)).astype(np.uint8)
    input_dict = {
        "a": a,
        "axis": 1,
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.median_1"] = median_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.median_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.median_1'.")


check_valid('jax.numpy.median', generated_inputs['jax.numpy.median_1'], lib="jax", suffix=1)

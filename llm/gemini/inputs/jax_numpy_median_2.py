
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def median_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis (0,), no keepdims
    a = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with negative and positive values, axis (1,)
    a = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (1,),
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, axis (0,), keepdims=True
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, multi-axis reduction, keepdims=False
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array with large integers, axis (-1,)
    a = np.random.randint(-1000, 1000, size=(3, 3, 3)).astype(np.int64)
    input_dict = {
        "a": a,
        "axis": (-1,),
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array, keepdims=True, overwrite_input=False
    a = np.random.uniform(-5.0, 5.0, (4, 6)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, multi-axis reduction (1, 3), keepdims=True
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int32 array with sorted negative values
    a = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": (-1,),
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array, full reduction using axis tuple, keepdims=True
    a = np.random.randn(10, 10).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int32 array reducing all three dimensions, keepdims=False
    a = np.random.randint(0, 100, size=(2, 2, 2)).astype(np.int32)
    input_dict = {
        "a": a,
        "axis": (0, 1, 2),
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.median_2"] = median_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.median_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.median_2'.")


check_valid('jax.numpy.median', generated_inputs['jax.numpy.median_2'], lib="jax", suffix=2)

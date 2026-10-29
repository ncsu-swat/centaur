
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmedian_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs
    a = np.array([1.0, np.nan, 3.0, 7.0, 5.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "keepdims": False,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with NaNs, axis 1, keepdims True
    a = np.array(
        [[2.0, np.nan, 7.0, np.nan], [np.nan, 5.0, 9.0, 2.0], [6.0, 1.0, np.nan, 3.0]],
        dtype=np.float32,
    )
    input_dict = {
        "a": a,
        "axis": (1,),
        "keepdims": True,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array with negative values and NaNs, axis 0
    a = np.array(
        [[-10.5, np.nan, -2.0], [3.5, -4.0, np.nan], [np.nan, 0.0, 8.2]],
        dtype=np.float64,
    )
    input_dict = {
        "a": a,
        "axis": (0,),
        "keepdims": False,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array reducing over all axes (0, 1) with keepdims True
    a = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "keepdims": True,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array reducing over multiple axes
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "keepdims": False,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 array with negative axis
    a = np.linspace(-5.0, 5.0, 30, dtype=np.float64).reshape(2, 5, 3)
    a[0, 0, 0] = np.nan
    input_dict = {
        "a": a,
        "axis": (-1,),
        "keepdims": True,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    a = np.ones((2, 3, 2, 4), dtype=np.float32)
    a[1, :, 0, :] = np.nan
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "keepdims": False,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 array (no NaNs)
    a = np.array([[10, 20, 30], [5, 15, 25]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "keepdims": True,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array reducing along all axes
    a = np.array([[[np.nan, 2.0], [3.0, 4.0]], [[5.0, np.nan], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (0, 1, 2),
        "keepdims": False,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array with all NaNs
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": (0,),
        "keepdims": True,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float32 array with negative axis indexing
    a = np.array([[-3.0, -1.0, 0.0], [np.nan, 2.0, 4.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (-2,),
        "keepdims": False,
        "overwrite_input": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanmedian_2"] = nanmedian_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmedian_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmedian_2'.")


check_valid('jax.numpy.nanmedian', generated_inputs['jax.numpy.nanmedian_2'], lib="jax", suffix=2)

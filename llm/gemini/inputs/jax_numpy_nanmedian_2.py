
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_nanmedian_inputs():
    list_of_inputs = []

    # Input 1: 2D array with NaNs, axis=(0,), keepdims=False, overwrite_input=False
    a1 = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, np.nan]], dtype=np.float32)
    input_dict = {
        "a": a1,
        "axis": (0,),
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array without NaNs, axis=(1,), keepdims=True, overwrite_input=False
    a2 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    input_dict = {
        "a": a2,
        "axis": (1,),
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with NaNs, axis=(0, 2), keepdims=False, overwrite_input=False
    a3 = np.random.randn(2, 3, 4).astype(np.float32)
    a3[0, 1, 2] = np.nan
    a3[1, 2, 0] = np.nan
    input_dict = {
        "a": a3,
        "axis": (0, 2),
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with NaNs, axis=(0,), keepdims=True, overwrite_input=False
    a4 = np.array([1.5, np.nan, -2.5, 4.0, np.nan], dtype=np.float32)
    input_dict = {
        "a": a4,
        "axis": (0,),
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array with NaNs, axis=(1, 3), keepdims=False, overwrite_input=False
    a5 = np.random.randn(2, 2, 2, 2).astype(np.float64)
    a5[a5 > 1.0] = np.nan
    input_dict = {
        "a": a5,
        "axis": (1, 3),
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with negative values and NaNs, axis=(-1,), keepdims=True, overwrite_input=False
    a6 = np.array([[-10.0, np.nan, -3.0], [np.nan, -5.0, -6.0]], dtype=np.float32)
    input_dict = {
        "a": a6,
        "axis": (-1,),
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with one slice containing all NaNs, axis=(1,), keepdims=False, overwrite_input=False
    a7 = np.zeros((2, 3, 2), dtype=np.float32)
    a7[:, :, :] = np.nan
    a7[0, 1, 0] = 5.0
    input_dict = {
        "a": a7,
        "axis": (1,),
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, large size, float64, axis=(0,), keepdims=True, overwrite_input=False
    a8 = np.random.randn(50, 100).astype(np.float64)
    a8[a8 < -1.0] = np.nan
    input_dict = {
        "a": a8,
        "axis": (0,),
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, axis=(2,), keepdims=False, overwrite_input=False
    a9 = np.random.randn(3, 3, 3).astype(np.float32)
    a9[0, 0, :] = np.nan
    input_dict = {
        "a": a9,
        "axis": (2,),
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, small sizes, axis=(2, 4), keepdims=True, overwrite_input=False
    a10 = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    a10[0, 0, 0, 0, 0] = np.nan
    input_dict = {
        "a": a10,
        "axis": (2, 4),
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanmedian_2"] = jax_numpy_nanmedian_inputs()

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

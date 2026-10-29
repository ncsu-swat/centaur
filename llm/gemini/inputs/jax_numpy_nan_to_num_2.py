
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nan_to_num_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with nan, inf, -inf
    x = np.array([np.nan, 1.0, np.inf, -np.inf, 2.5], dtype=np.float32)
    copy_flag = True
    nan_val = np.array(0.0, dtype=np.float32)
    posinf_val = np.array(999.0, dtype=np.float32)
    neginf_val = np.array(-999.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 2: 2D float64 array with copy=False
    x = np.array([[np.nan, 2.0], [-np.inf, np.inf]], dtype=np.float64)
    copy_flag = False
    nan_val = np.array(-1.0, dtype=np.float64)
    posinf_val = np.array(1e10, dtype=np.float64)
    neginf_val = np.array(-1e10, dtype=np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 3: 3D float32 array
    x = np.array(
        [[[np.nan, 0.0], [np.inf, -np.inf]], [[1.0, 2.0], [np.nan, np.nan]]],
        dtype=np.float32,
    )
    copy_flag = True
    nan_val = np.array(0.0, dtype=np.float32)
    posinf_val = np.array(1e5, dtype=np.float32)
    neginf_val = np.array(-1e5, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 4: 0D scalar float32 array
    x = np.array(np.nan, dtype=np.float32)
    copy_flag = True
    nan_val = np.array(42.0, dtype=np.float32)
    posinf_val = np.array(100.0, dtype=np.float32)
    neginf_val = np.array(-100.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 5: 1D float16 array
    x = np.array([-np.inf, np.nan, 0.5, np.inf], dtype=np.float16)
    copy_flag = False
    nan_val = np.array(0.0, dtype=np.float16)
    posinf_val = np.array(65504.0, dtype=np.float16)
    neginf_val = np.array(-65504.0, dtype=np.float16)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 6: 2D int32 array (non-inexact dtype returned unmodified)
    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    copy_flag = True
    nan_val = np.array(0, dtype=np.int32)
    posinf_val = np.array(2147483647, dtype=np.int32)
    neginf_val = np.array(-2147483648, dtype=np.int32)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 7: 4D float32 array
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    x[0, 0, 0, 0] = np.nan
    x[0, 1, 0, 1] = np.inf
    x[1, 0, 1, 0] = -np.inf
    copy_flag = True
    nan_val = np.array(5.0, dtype=np.float32)
    posinf_val = np.array(1000.0, dtype=np.float32)
    neginf_val = np.array(-1000.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 8: Complex64 1D array
    x = np.array(
        [complex(np.nan, 1.0), complex(2.0, np.inf), complex(-np.inf, np.nan)],
        dtype=np.complex64,
    )
    copy_flag = False
    nan_val = np.array(0.0, dtype=np.float32)
    posinf_val = np.array(1e6, dtype=np.float32)
    neginf_val = np.array(-1e6, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 9: 2D float32 array with broadcastable 1D replacement tensors
    x = np.array([[np.nan, np.inf], [-np.inf, np.nan]], dtype=np.float32)
    copy_flag = True
    nan_val = np.array([10.0, 20.0], dtype=np.float32)
    posinf_val = np.array([100.0, 200.0], dtype=np.float32)
    neginf_val = np.array([-100.0, -200.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    # Input 10: 1D float64 with all finite elements
    x = np.array([1.23, -4.56, 7.89, -10.11], dtype=np.float64)
    copy_flag = True
    nan_val = np.array(0.0, dtype=np.float64)
    posinf_val = np.array(1e300, dtype=np.float64)
    neginf_val = np.array(-1e300, dtype=np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "copy": copy_flag,
            "nan": nan_val,
            "posinf": posinf_val,
            "neginf": neginf_val,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nan_to_num_2"] = nan_to_num_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nan_to_num_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nan_to_num_2'.")


check_valid('jax.numpy.nan_to_num', generated_inputs['jax.numpy.nan_to_num_2'], lib="jax", suffix=2)

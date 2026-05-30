
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nan_to_num_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, copy=True
    x = np.array([1.0, np.nan, 2.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    copy_val = True
    nan_val = np.array(0.0, dtype=np.float32)
    posinf_val = np.array(100.0, dtype=np.float32)
    neginf_val = np.array(-100.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 2: 2D array, float64, copy=False
    x = np.array([[np.nan, 2.0], [np.inf, -np.inf]], dtype=np.float64)
    copy_val = False
    nan_val = np.array(-1.0, dtype=np.float64)
    posinf_val = np.array(1e10, dtype=np.float64)
    neginf_val = np.array(-1e10, dtype=np.float64)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 3: 3D array, float32, random initialization
    x = np.random.randn(2, 2, 2).astype(np.float32)
    x[0, 0, 0] = np.nan
    x[1, 1, 1] = np.inf
    copy_val = True
    nan_val = np.array(99.0, dtype=np.float32)
    posinf_val = np.array(999.0, dtype=np.float32)
    neginf_val = np.array(-999.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 4: 1D array, float16, copy=True
    x = np.array([np.nan, np.inf, -np.inf], dtype=np.float16)
    copy_val = True
    nan_val = np.array(0.0, dtype=np.float16)
    posinf_val = np.array(500.0, dtype=np.float16)
    neginf_val = np.array(-500.0, dtype=np.float16)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 5: 4D array, float32, copy=False
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x[0, 0, 0, 0] = np.nan
    x[1, 1, 1, 1] = -np.inf
    copy_val = False
    nan_val = np.array(-9999.0, dtype=np.float32)
    posinf_val = np.array(9999.0, dtype=np.float32)
    neginf_val = np.array(-9999.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 6: 0D array (scalar array)
    x = np.array(np.nan, dtype=np.float32)
    copy_val = True
    nan_val = np.array(1.23, dtype=np.float32)
    posinf_val = np.array(4.56, dtype=np.float32)
    neginf_val = np.array(-4.56, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 7: Array without any NaNs or Infs
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    copy_val = False
    nan_val = np.array(0.0, dtype=np.float32)
    posinf_val = np.array(0.0, dtype=np.float32)
    neginf_val = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 8: 5D array, float64
    x = np.ones((2, 2, 2, 2, 2), dtype=np.float64) * np.nan
    copy_val = True
    nan_val = np.array(-5.0, dtype=np.float64)
    posinf_val = np.array(5.0, dtype=np.float64)
    neginf_val = np.array(-5.0, dtype=np.float64)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 9: 1D array, large substitution values
    x = np.array([np.nan, 10.0, np.inf], dtype=np.float32)
    copy_val = True
    nan_val = np.array(42.0, dtype=np.float32)
    posinf_val = np.array(1000.0, dtype=np.float32)
    neginf_val = np.array(-1000.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

    # Input 10: Array with matched-shape replacement tensors
    x = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    copy_val = False
    nan_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    posinf_val = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    neginf_val = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "copy": copy_val,
        "nan": nan_val,
        "posinf": posinf_val,
        "neginf": neginf_val
    })

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

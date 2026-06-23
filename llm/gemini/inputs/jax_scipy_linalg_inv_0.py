
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_linalg_inv_inputs():
    list_of_inputs = []

    # All inputs use shape (2, 2) and float32 to minimize JAX compilation overhead.
    
    # 1
    list_of_inputs.append({
        "a": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "overwrite_a": False,
        "check_finite": True
    })

    # 2
    list_of_inputs.append({
        "a": np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        "overwrite_a": True,
        "check_finite": True
    })

    # 3
    list_of_inputs.append({
        "a": np.array([[3.0, -1.0], [-1.0, 3.0]], dtype=np.float32),
        "overwrite_a": False,
        "check_finite": False
    })

    # 4
    list_of_inputs.append({
        "a": np.array([[1.5, 0.5], [0.5, 1.5]], dtype=np.float32),
        "overwrite_a": True,
        "check_finite": False
    })

    # 5
    list_of_inputs.append({
        "a": np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=np.float32),
        "overwrite_a": False,
        "check_finite": True
    })

    # 6
    list_of_inputs.append({
        "a": np.array([[5.0, 2.0], [2.0, 5.0]], dtype=np.float32),
        "overwrite_a": True,
        "check_finite": True
    })

    # 7
    list_of_inputs.append({
        "a": np.array([[10.0, 1.0], [1.0, 10.0]], dtype=np.float32),
        "overwrite_a": False,
        "check_finite": False
    })

    # 8
    list_of_inputs.append({
        "a": np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32),
        "overwrite_a": True,
        "check_finite": False
    })

    # 9
    list_of_inputs.append({
        "a": np.array([[4.0, -2.0], [-2.0, 4.0]], dtype=np.float32),
        "overwrite_a": False,
        "check_finite": True
    })

    # 10
    list_of_inputs.append({
        "a": np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
        "overwrite_a": True,
        "check_finite": True
    })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.inv"] = jax_scipy_linalg_inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.inv'.")


check_valid('jax.scipy.linalg.inv', generated_inputs['jax.scipy.linalg.inv'], lib="jax", suffix=0)

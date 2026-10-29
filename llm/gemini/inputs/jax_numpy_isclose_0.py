
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isclose_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays with default tolerances and equal_nan=False
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.000001, 2.0, 3.00001], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-05,
            "atol": 1e-08,
            "equal_nan": False,
        }
    )

    # Input 2: 2D float64 arrays with negative values and custom tolerances
    a = np.array([[-1.0, 2.5], [-3.14, 0.0]], dtype=np.float64)
    b = np.array([[-1.0001, 2.5], [-3.1402, 1e-9]], dtype=np.float64)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-03,
            "atol": 1e-06,
            "equal_nan": False,
        }
    )

    # Input 3: Arrays with NaNs and equal_nan=True
    a = np.array([np.nan, 1.0, np.nan, 4.0], dtype=np.float32)
    b = np.array([np.nan, 1.0, 2.0, np.nan], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-05,
            "atol": 1e-08,
            "equal_nan": True,
        }
    )

    # Input 4: Arrays with Infs and NaNs, equal_nan=False
    a = np.array([np.inf, -np.inf, 1.0, np.nan], dtype=np.float64)
    b = np.array([np.inf, -np.inf, 1.0, np.nan], dtype=np.float64)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-04,
            "atol": 1e-05,
            "equal_nan": False,
        }
    )

    # Input 5: 3D float32 arrays
    a = np.ones((2, 3, 4), dtype=np.float32)
    b = np.ones((2, 3, 4), dtype=np.float32) + 1e-6
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-05,
            "atol": 1e-05,
            "equal_nan": True,
        }
    )

    # Input 6: Broadcasting between shapes (3, 1) and (1, 4)
    a = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-02,
            "atol": 0.1,
            "equal_nan": False,
        }
    )

    # Input 7: 0D scalar arrays
    a = np.array(42.0, dtype=np.float64)
    b = np.array(42.0000001, dtype=np.float64)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-06,
            "atol": 1e-06,
            "equal_nan": True,
        }
    )

    # Input 8: Integer arrays
    a = np.array([-10, 0, 10, 20], dtype=np.int32)
    b = np.array([-10, 1, 10, 21], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 0.1,
            "atol": 1.0,
            "equal_nan": False,
        }
    )

    # Input 9: Complex numbers
    a = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.0 - 1.0j], dtype=np.complex64)
    b = np.array([1.00001 + 2.0j, -3.0 + 4.00001j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 1e-04,
            "atol": 1e-04,
            "equal_nan": False,
        }
    )

    # Input 10: 4D float32 arrays with zero tolerance
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    b = np.zeros((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "rtol": 0.0,
            "atol": 0.0,
            "equal_nan": True,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.isclose"] = isclose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isclose'.")


check_valid('jax.numpy.isclose', generated_inputs['jax.numpy.isclose'], lib="jax", suffix=0)

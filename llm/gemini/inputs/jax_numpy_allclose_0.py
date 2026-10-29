
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def allclose_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, exact match
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "rtol": 1e-05,
                "atol": 1e-08,
                "equal_nan": False,
            }
        )
    )

    # Input 2: 2D float64 arrays with negative values and close elements
    a = np.array([[-1.0, 2.5], [3.3, -4.8]], dtype=np.float64)
    b = np.array([[-1.00001, 2.50001], [3.29999, -4.80002]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-04, "atol": 1e-05, "equal_nan": False}
        )
    )

    # Input 3: Broadcastable shapes (2D and 1D)
    a = np.ones((3, 4), dtype=np.float32)
    b = np.ones((4,), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-05, "atol": 1e-08, "equal_nan": False}
        )
    )

    # Input 4: 3D arrays with NaNs and equal_nan=True
    a = np.array([[[np.nan, 1.0], [2.0, 3.0]]], dtype=np.float32)
    b = np.array([[[np.nan, 1.0], [2.0, 3.0]]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-05, "atol": 1e-08, "equal_nan": True}
        )
    )

    # Input 5: 3D arrays with NaNs and equal_nan=False
    a = np.array([[[np.nan, 1.0], [2.0, 3.0]]], dtype=np.float32)
    b = np.array([[[np.nan, 1.0], [2.0, 3.0]]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-05, "atol": 1e-08, "equal_nan": False}
        )
    )

    # Input 6: Arrays containing infs
    a = np.array([np.inf, -np.inf, 0.0], dtype=np.float32)
    b = np.array([np.inf, -np.inf, 1e-9], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-05, "atol": 1e-08, "equal_nan": False}
        )
    )

    # Input 7: 0D scalar arrays
    a = np.array(5.0, dtype=np.float32)
    b = np.array(5.000001, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-04, "atol": 1e-06, "equal_nan": False}
        )
    )

    # Input 8: Integer arrays
    a = np.array([[10, 20], [30, 40]], dtype=np.int32)
    b = np.array([[10, 20], [30, 41]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-01, "atol": 1.0, "equal_nan": False}
        )
    )

    # Input 9: Complex numbers
    a = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    b = np.array([1.000001 + 2.0j, -3.0 - 4.000001j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-04, "atol": 1e-05, "equal_nan": False}
        )
    )

    # Input 10: 4D arrays with large tolerances
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    b = np.full((2, 2, 2, 2), 0.05, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-01, "atol": 0.1, "equal_nan": True}
        )
    )

    # Input 11: 1D arrays with float16 dtype
    a = np.array([-100.0, 0.0, 100.0], dtype=np.float16)
    b = np.array([-100.1, 0.0, 99.9], dtype=np.float16)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "b": b, "rtol": 1e-02, "atol": 0.2, "equal_nan": False}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.allclose"] = allclose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.allclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.allclose'.")


check_valid('jax.numpy.allclose', generated_inputs['jax.numpy.allclose'], lib="jax", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isclose_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays, identical
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    })

    # Input 2: 2D float64 arrays, small difference within atol
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[1.000000001, 2.0], [3.0, 3.999999999]], dtype=np.float64)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    })

    # Input 3: Negative values, 3D float32
    a = np.array([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=np.float32)
    b = np.array([[[-1.00001, -2.0], [-3.0, -4.00001]]], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-04,
        "atol": 1e-05,
        "equal_nan": False
    })

    # Input 4: float16 arrays, 1D
    a = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    b = np.array([0.501, 1.499, 2.5], dtype=np.float16)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-02,
        "atol": 1e-02,
        "equal_nan": True
    })

    # Input 5: NaNs, equal_nan=True
    a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    b = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": True
    })

    # Input 6: NaNs, equal_nan=False
    a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    b = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    })

    # Input 7: Infinities
    a = np.array([np.inf, -np.inf, 1.0], dtype=np.float32)
    b = np.array([np.inf, -np.inf, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    })

    # Input 8: High dimensional 4D arrays, small random differences
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    b = a + np.random.normal(0, 1e-6, a.shape).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-04,
        "atol": 1e-05,
        "equal_nan": False
    })

    # Input 9: Large absolute difference requiring custom atol
    a = np.array([100.0, 200.0], dtype=np.float32)
    b = np.array([105.0, 195.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-09,
        "atol": 10.0,
        "equal_nan": False
    })

    # Input 10: Large values requiring custom rtol
    a = np.array([1e6, 2e6], dtype=np.float32)
    b = np.array([1.01e6, 1.99e6], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 0.02,
        "atol": 1e-08,
        "equal_nan": False
    })

    # Input 11: Scalar-like 0D arrays
    a = np.array(1.0001, dtype=np.float32)
    b = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "b": b,
        "rtol": 1e-03,
        "atol": 1e-03,
        "equal_nan": False
    })

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

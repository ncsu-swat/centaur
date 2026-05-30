
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanquantile_inputs():
    list_of_inputs = []

    def make_array_with_nans(shape, dtype=np.float32):
        arr = np.random.randn(*shape).astype(dtype)
        mask = np.random.rand(*shape) > 0.8
        arr[mask] = np.nan
        return arr

    # Input 1
    a = make_array_with_nans((5,))
    q = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.1, 1.0, size=(5,)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 2
    a = make_array_with_nans((3, 4))
    q = np.array([0.5], dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.1, 1.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 3
    a = make_array_with_nans((2, 3, 4))
    q = np.array([0.1], dtype=np.float32)
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.1, 1.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 4
    a = make_array_with_nans((10,))
    q = np.array([0.1, 0.9], dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.1, 1.0, size=(10,)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 5
    a = make_array_with_nans((5, 5), dtype=np.float64)
    q = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.1, 1.0, size=(5, 5)).astype(np.float64)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 6
    a = make_array_with_nans((4, 4, 4))
    q = np.array([0.0, 1.0], dtype=np.float32)
    axis = (2,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.1, 1.0, size=(4, 4, 4)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 7
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    q = np.array([0.5], dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 8
    a = make_array_with_nans((2, 10))
    q = np.array([0.3, 0.7], dtype=np.float32)
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.1, 1.0, size=(2, 10)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 9
    a = make_array_with_nans((2, 2, 2, 2))
    q = np.array([0.5], dtype=np.float32)
    axis = (3,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.1, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 10
    a = make_array_with_nans((3, 2, 5))
    q = np.array([0.25, 0.75], dtype=np.float32)
    axis = (2,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.1, 1.0, size=(3, 2, 5)).astype(np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    return list_of_inputs

generated_inputs["jax.numpy.nanquantile_2"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_2'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_2'], lib="jax", suffix=2)

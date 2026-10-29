
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def quantile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, single quantile
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array(0.5, dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 2: 1D array with multiple quantiles, keepdims=True
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 3: 2D array, axis=0
    a = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    q = np.array([0.2, 0.8], dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((3, 3), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 4: 2D array, axis=1
    a = np.array(
        [[10.0, 20.0, 30.0, 40.0], [5.0, 15.0, 25.0, 35.0]], dtype=np.float32
    )
    q = np.array([0.5], dtype=np.float32)
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array(
        [[1.0, 2.0, 1.0, 2.0], [2.0, 1.0, 2.0, 1.0]], dtype=np.float32
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 5: 2D array, axis=(0, 1) - multiple axes reduction
    a = np.arange(12, dtype=np.float32).reshape(3, 4) - 6.0
    q = np.array(0.3, dtype=np.float32)
    axis = (0, 1)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 6: 3D array, negative axis index
    a = np.random.RandomState(42).randn(2, 3, 4).astype(np.float32)
    q = np.array([0.1, 0.9], dtype=np.float32)
    axis = (-1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 7: 3D array, multiple axes reduction, keepdims=True
    a = np.random.RandomState(7).randn(4, 5, 6).astype(np.float64)
    q = np.array([0.5], dtype=np.float64)
    axis = (0, 2)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((4, 5, 6), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 8: Integer input array promoted in quantile
    a = np.array([-5, -3, -1, 1, 3, 5], dtype=np.int32)
    q = np.array([0.0, 1.0], dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 9: 4D array with axis reduction on middle axes
    a = np.random.RandomState(123).uniform(-10.0, 10.0, size=(2, 3, 4, 2)).astype(np.float32)
    q = np.array([0.25, 0.75], dtype=np.float32)
    axis = (1, 2)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((2, 3, 4, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 10: 2D array, non-uniform weights along reduced axis
    a = np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]], dtype=np.float64)
    q = np.array(0.7, dtype=np.float64)
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([[0.1, 0.3, 0.6], [0.5, 0.3, 0.2]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.quantile_2"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_2'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_2'], lib="jax", suffix=2)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanpercentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs, scalar q, axis 0
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    q = np.array(50.0, dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, multiple quantiles, axis 0, keepdims True
    a = np.array(
        [[10.0, np.nan, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
        dtype=np.float32,
    )
    q = np.array([25.0, 75.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array with NaNs, axis 1, keepdims False
    a = np.array(
        [[1.5, 2.5, np.nan, 4.5], [np.nan, 6.5, 7.5, 8.5]], dtype=np.float64
    )
    q = np.array([10.0, 50.0, 90.0], dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, negative axis, keepdims True, q scalar
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    q = np.array(0.0, dtype=np.float32)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array with extreme percentiles (0 and 100)
    a = np.array([-10.0, 0.0, 10.0, np.nan, 20.0], dtype=np.float64)
    q = np.array([0.0, 100.0], dtype=np.float64)
    weights = np.array([0.5, 1.0, 1.5, 2.0, 0.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 1, keepdims False
    a = np.random.RandomState(42).randn(3, 5, 2).astype(np.float32)
    a[1, 2, 0] = np.nan
    q = np.array([50.0], dtype=np.float32)
    weights = np.ones((3, 5, 2), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with negative values and negative axis -2
    a = np.array([[-5.0, -10.0], [np.nan, -2.0], [-1.0, -8.0]], dtype=np.float32)
    q = np.array([33.3, 66.6], dtype=np.float32)
    weights = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": -2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, axis 2, keepdims False
    a = np.ones((2, 2, 4, 3), dtype=np.float32)
    a[0, 0, 1, 0] = np.nan
    q = np.array(100.0, dtype=np.float32)
    weights = np.ones((2, 2, 4, 3), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array, axis 1, keepdims True, high percentiles
    a = np.array([[100.0, 200.0, np.nan], [400.0, 500.0, 600.0]], dtype=np.float64)
    q = np.array([5.0, 95.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array, multiple quantiles, axis 0, keepdims True
    a = np.array([np.nan, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
    q = np.array([12.5, 37.5, 62.5, 87.5], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanpercentile_1"] = nanpercentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanpercentile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanpercentile_1'.")


check_valid('jax.numpy.nanpercentile', generated_inputs['jax.numpy.nanpercentile_1'], lib="jax", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def percentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, median, keepdims=False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array(50.0, dtype=np.float32)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 2: 2D float32 array along axis 1, multiple percentiles, keepdims=True
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    q = np.array([25.0, 75.0], dtype=np.float32)
    axis = 1
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 3: 2D float64 array with negative values, axis 0
    a = np.array([[-5.0, 2.5], [1.0, -3.2], [4.4, 0.0]], dtype=np.float64)
    q = np.array([10.0, 50.0, 90.0], dtype=np.float64)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 4: 3D float32 array, negative axis index -1
    a = np.random.randn(2, 3, 4).astype(np.float32)
    q = np.array(75.0, dtype=np.float32)
    axis = -1
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.1, 1.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 5: 1D int32 array, 0th and 100th percentiles
    a = np.array([-10, 0, 5, 20, 15], dtype=np.int32)
    q = np.array([0.0, 100.0], dtype=np.float32)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 6: 3D float32 array along middle axis 1
    a = np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(3, 2, 4)
    q = np.array([50.0], dtype=np.float32)
    axis = 1
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((3, 2, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 7: 4D float32 array, axis 2, keepdims=True
    a = np.random.randn(2, 2, 3, 2).astype(np.float32)
    q = np.array([25.0, 50.0, 75.0], dtype=np.float32)
    axis = 2
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((2, 2, 3, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 8: 2D integer array, axis 0
    a = np.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int64
    )
    q = np.array(33.3, dtype=np.float64)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.array(
        [[1.0, 1.5, 2.0], [2.0, 1.0, 1.0], [3.0, 2.5, 1.5]], dtype=np.float64
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 9: 1D array, scalar q as 0D tensor, keepdims=True
    a = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    q = np.array(0.0, dtype=np.float32)
    axis = -1
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([0.5, 1.5, 2.5, 0.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 10: 3D float64 array, axis 0, multiple quantiles
    a = np.random.uniform(-10.0, 10.0, size=(5, 3, 2)).astype(np.float64)
    q = np.array([5.0, 95.0], dtype=np.float64)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.exponential(scale=1.0, size=(5, 3, 2)).astype(
        np.float64
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.percentile_1"] = percentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.percentile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.percentile_1'.")


check_valid('jax.numpy.percentile', generated_inputs['jax.numpy.percentile_1'], lib="jax", suffix=1)

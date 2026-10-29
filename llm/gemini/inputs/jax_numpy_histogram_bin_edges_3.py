
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard positive range
    a = np.array([1.0, 2.5, 3.2, 4.8, 5.0], dtype=np.float32)
    bins = 5
    range_param = [0.0, 10.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 2: 2D int32 array, symmetric negative to positive range
    a = np.array([[-5, 2], [3, -1], [0, 4]], dtype=np.int32)
    bins = 10
    range_param = [-10.0, 10.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 3: 3D float64 array
    a = np.linspace(1.0, 100.0, 24, dtype=np.float64).reshape((2, 3, 4))
    bins = 20
    range_param = [1.0, 100.0]
    weights = np.ones_like(a, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 4: 1D int64 array with all negative values
    a = np.array([-50, -40, -30, -20, -10], dtype=np.int64)
    bins = 3
    range_param = [-50.0, 0.0]
    weights = np.array([1, 2, 1, 1, 2], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 5: Single element 1D array with single bin
    a = np.array([0.5], dtype=np.float32)
    bins = 1
    range_param = [0.0, 1.0]
    weights = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 6: 4D float32 array
    a = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    bins = 8
    range_param = [-1.0, 1.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 7: 2D uint8 array
    a = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    bins = 16
    range_param = [0, 256]
    weights = np.ones_like(a, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 8: 1D float16 array
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    bins = 4
    range_param = [-2.5, 2.5]
    weights = np.ones_like(a, dtype=np.float16)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 9: Large values float64 array with many bins
    a = np.array([1200.0, 2500.0, 3100.0, 4800.0], dtype=np.float64)
    bins = 50
    range_param = [1000.0, 5000.0]
    weights = np.ones_like(a, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    # Input 10: 1D float32 array with fractional range
    a = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7], dtype=np.float32)
    bins = 7
    range_param = [0.5, 9.5]
    weights = np.array([0.5, 1.0, 1.5, 2.0, 1.0, 0.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "bins": bins, "range": range_param, "weights": weights}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogram_bin_edges_3"] = (
    histogram_bin_edges_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_bin_edges_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_bin_edges_3'.")


check_valid('jax.numpy.histogram_bin_edges', generated_inputs['jax.numpy.histogram_bin_edges_3'], lib="jax", suffix=3)

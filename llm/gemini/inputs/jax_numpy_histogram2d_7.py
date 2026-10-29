
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays with 0D int array for bins
    x = np.array([1.0, 2.5, 3.0, 5.5, 7.0, 8.5], dtype=np.float32)
    y = np.array([2.0, 3.5, 4.0, 6.0, 7.5, 9.0], dtype=np.float32)
    bins = np.array(5, dtype=np.int32)
    range_param = ((0.0, 10.0), (0.0, 10.0))
    weights = np.ones(6, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 2: float64 arrays with negative values, 1D bins tensor, density=True
    x = np.array([-4.0, -2.0, 0.0, 1.5, 3.0], dtype=np.float64)
    y = np.array([-2.5, -1.0, 0.5, 1.0, 2.5], dtype=np.float64)
    bins = np.array([4, 6], dtype=np.int32)
    range_param = ((-5.0, 5.0), (-3.0, 3.0))
    weights = np.array([0.5, 1.0, 1.5, 2.0, 0.5], dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 3: int32 data, 0D bins tensor, float32 weights
    x = np.array([1, 4, 8, 12, 15, 18], dtype=np.int32)
    y = np.array([3, 6, 9, 11, 14, 17], dtype=np.int32)
    bins = np.array(10, dtype=np.int64)
    range_param = ((0.0, 20.0), (0.0, 20.0))
    weights = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 2.5], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 4: Wider range and asymmetric bins
    x = np.linspace(-90.0, 90.0, 20, dtype=np.float32)
    y = np.linspace(-40.0, 40.0, 20, dtype=np.float32)
    bins = np.array([8, 12], dtype=np.int32)
    range_param = ((-100.0, 100.0), (-50.0, 50.0))
    weights = np.full(20, 0.5, dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 5: Small dataset within [0, 1] range
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    y = np.array([0.2, 0.4, 0.6, 0.8, 0.95], dtype=np.float64)
    bins = np.array([2, 3], dtype=np.int64)
    range_param = ((0.0, 1.0), (0.0, 1.0))
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 6: int64 data and int64 weights
    x = np.array([-8, -4, 0, 3, 7], dtype=np.int64)
    y = np.array([-5, -1, 2, 4, 8], dtype=np.int64)
    bins = np.array(8, dtype=np.int32)
    range_param = ((-10.0, 10.0), (-10.0, 10.0))
    weights = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 7: Larger sample size with density=True
    np.random.seed(42)
    x = np.random.uniform(-1.0, 1.0, size=200).astype(np.float32)
    y = np.random.uniform(-1.0, 1.0, size=200).astype(np.float32)
    bins = np.array([15, 15], dtype=np.int32)
    range_param = ((-1.0, 1.0), (-1.0, 1.0))
    weights = np.random.uniform(0.1, 2.0, size=200).astype(np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 8: Few bins, non-zero minimum boundaries
    x = np.array([1.2, 2.4, 3.1, 4.8], dtype=np.float32)
    y = np.array([2.5, 3.6, 4.2, 5.9], dtype=np.float32)
    bins = np.array(3, dtype=np.int32)
    range_param = ((1.0, 5.0), (2.0, 6.0))
    weights = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 9: float64 with different range per dimension
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    bins = np.array([7, 5], dtype=np.int32)
    range_param = ((-2.5, 2.5), (0.0, 5.0))
    weights = np.array([1.2, 0.8, 1.5, 2.0, 0.3], dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 10: Mixed negative/positive sub-ranges
    x = np.array([-9.0, -6.0, -4.0, -1.0], dtype=np.float32)
    y = np.array([1.0, 3.0, 7.0, 9.0], dtype=np.float32)
    bins = np.array(6, dtype=np.int64)
    range_param = ((-10.0, 0.0), (0.0, 10.0))
    weights = np.array([2.0, 1.5, 0.5, 3.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogram2d_7"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_7'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_7'], lib="jax", suffix=7)

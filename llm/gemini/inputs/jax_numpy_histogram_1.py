
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard float32, density=False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bins = 5
    range_val = (0.0, 6.0)
    weights = np.ones_like(a, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 2: 1D array with negative values, density=True
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    bins = 10
    range_val = (-15.0, 15.0)
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 3: 2D array, integer data
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    bins = 6
    range_val = (0, 12)
    weights = np.ones((3, 4), dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 4: 3D array, density=True
    a = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    bins = 8
    range_val = (-1.0, 1.0)
    weights = np.abs(a)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 5: Broadcasted weights (1D weights for 2D data)
    a = np.ones((4, 5), dtype=np.float32)
    bins = 4
    range_val = (0.0, 2.0)
    weights = np.full((4, 5), 0.5, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 6: Large 1D array from normal distribution
    np.random.seed(42)
    a = np.random.randn(100).astype(np.float64)
    bins = 20
    range_val = (-3.0, 3.0)
    weights = np.random.rand(100).astype(np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 7: Single element array
    a = np.array([42.0], dtype=np.float32)
    bins = 1
    range_val = (40.0, 50.0)
    weights = np.array([1.5], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 8: Int64 input data with large range
    a = np.array([-1000, 0, 500, 1000], dtype=np.int64)
    bins = 4
    range_val = (-1000, 1000)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 9: 4D tensor input
    a = np.random.uniform(0.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    bins = 5
    range_val = (0.0, 10.0)
    weights = np.ones((2, 2, 2, 2), dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 10: Float16 precision
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    bins = 3
    range_val = (0.0, 1.0)
    weights = np.array([1.0, 1.0, 2.0, 2.0, 1.0], dtype=np.float16)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogram_1"] = histogram_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_1'.")


check_valid('jax.numpy.histogram', generated_inputs['jax.numpy.histogram_1'], lib="jax", suffix=1)

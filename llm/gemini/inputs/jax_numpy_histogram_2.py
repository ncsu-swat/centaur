
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, 1D bin edges, density=False
    a = np.array([0.5, 1.2, 2.3, 3.8, 4.1, 5.0], dtype=np.float32)
    bins = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    weights = np.array([1.0, 0.5, 1.0, 2.0, 1.5, 0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (0.0, 6.0),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, 0D bin count, density=True
    a = np.array([-2.5, -1.0, 0.0, 1.5, 2.5, 3.0], dtype=np.float64)
    bins = np.array(5, dtype=np.int32)
    weights = np.ones_like(a, dtype=np.float64)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (-3.0, 4.0),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, 1D non-uniform bin edges, density=False
    a = np.random.uniform(0.0, 10.0, size=(4, 5)).astype(np.float32)
    bins = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(4, 5)).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (0.0, 10.0),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, 0D bin count, density=True
    a = np.random.normal(loc=0.0, scale=1.0, size=(2, 3, 4)).astype(np.float32)
    bins = np.array(8, dtype=np.int32)
    weights = np.random.uniform(0.5, 1.5, size=(2, 3, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (-3.0, 3.0),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int32 array, 1D bin edges, density=False
    a = np.array([1, 2, 2, 3, 5, 7, 8, 9, 10], dtype=np.int32)
    bins = np.array([0.0, 3.0, 6.0, 9.0, 12.0], dtype=np.float32)
    weights = np.array(
        [1.0, 2.0, 1.0, 0.5, 1.5, 2.0, 1.0, 1.0, 0.5], dtype=np.float32
    )
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (0.0, 12.0),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D negative to positive values, 1D bin edges, density=True
    a = np.linspace(-5.0, 5.0, 20, dtype=np.float32)
    bins = np.linspace(-5.0, 5.0, 6, dtype=np.float32)
    weights = np.ones(20, dtype=np.float32)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (-5.0, 5.0),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array, 0D bin count, density=False
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    bins = np.array(6, dtype=np.int64)
    weights = np.full((3, 4), 2.0, dtype=np.float64)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (0.0, 12.0),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, 1D bin edges, density=False
    a = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 3)).astype(np.float32)
    bins = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    weights = np.ones((2, 2, 2, 3), dtype=np.float32)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (-1.0, 1.0),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D large array, 0D bin count, density=True
    a = np.random.randn(100).astype(np.float32)
    bins = np.array(15, dtype=np.int32)
    weights = np.random.uniform(0.0, 1.0, size=100).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (-4.0, 4.0),
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int64 array, 1D bin edges, density=False
    a = np.array([[-10, 0, 10], [20, 30, 40]], dtype=np.int64)
    bins = np.array([-15.0, 0.0, 15.0, 30.0, 45.0], dtype=np.float64)
    weights = np.array([[1.0, 2.0, 1.0], [0.5, 1.5, 2.5]], dtype=np.float64)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (-15.0, 45.0),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D single-element array, 0D bin count, density=False
    a = np.array([4.2], dtype=np.float32)
    bins = np.array(3, dtype=np.int32)
    weights = np.array([1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "bins": bins,
        "range": (0.0, 10.0),
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.histogram_2"] = histogram_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_2'.")


check_valid('jax.numpy.histogram', generated_inputs['jax.numpy.histogram_2'], lib="jax", suffix=2)

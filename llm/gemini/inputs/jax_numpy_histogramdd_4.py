
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogramdd_inputs():
    list_of_inputs = []

    # Input 1, valid - 2D data with float32
    sample = np.random.randn(100, 2).astype(np.float32)
    bins = (10, 10)
    range_val = ((-3.0, 3.0), (-3.0, 3.0))
    weights = np.ones(100, dtype=np.float32)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid - 3D data with float32 and density=True
    sample = np.random.randn(200, 3).astype(np.float32)
    bins = (5, 5, 5)
    range_val = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))
    weights = np.random.rand(200).astype(np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid - 1D data with float64
    sample = np.random.randn(50, 1).astype(np.float64)
    bins = (8,)
    range_val = ((-4.0, 4.0),)
    weights = np.random.rand(50).astype(np.float64)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid - 4D uniform distribution
    sample = np.random.uniform(-5.0, 5.0, (150, 4)).astype(np.float32)
    bins = (6, 6, 6, 6)
    range_val = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))
    weights = np.ones(150, dtype=np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - 2D data with float64 and custom weights range
    sample = np.random.randn(80, 2).astype(np.float64)
    bins = (12, 12)
    range_val = ((-2.5, 2.5), (-2.5, 2.5))
    weights = np.random.uniform(0.5, 1.5, 80).astype(np.float64)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - 3D data with asymmetric bin sizes
    sample = np.random.randn(300, 3).astype(np.float32)
    bins = (4, 8, 4)
    range_val = ((-3.0, 3.0), (-1.0, 1.0), (-2.0, 2.0))
    weights = np.random.randn(300).astype(np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid - 2D data with many bins
    sample = np.random.randn(120, 2).astype(np.float32)
    bins = (15, 10)
    range_val = ((-4.0, 4.0), (-4.0, 4.0))
    weights = np.random.rand(120).astype(np.float32)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid - 5D sparse binning
    sample = np.random.randn(50, 5).astype(np.float32)
    bins = (3, 3, 3, 3, 3)
    range_val = ((-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0))
    weights = np.ones(50, dtype=np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid - 2D uniform distribution with positive range
    sample = np.random.uniform(0.0, 10.0, (100, 2)).astype(np.float32)
    bins = (5, 5)
    range_val = ((0.0, 10.0), (0.0, 10.0))
    weights = np.random.rand(100).astype(np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid - 3D data with float64 and density=False
    sample = np.random.randn(150, 3).astype(np.float64)
    bins = (10, 10, 10)
    range_val = ((-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0))
    weights = np.random.rand(150).astype(np.float64)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogramdd_4"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_4'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_4'], lib="jax", suffix=4)

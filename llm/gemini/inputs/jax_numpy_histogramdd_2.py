
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogramdd_inputs():
    list_of_inputs = []

    # Input 1: 2D standard normal, 5x5 bins, range -3 to 3, uniform weights, density False
    sample = np.random.randn(100, 2).astype(np.float32)
    bins = np.array([5, 5], dtype=np.int32)
    range_val = [[-3.0, 3.0], [-3.0, 3.0]]
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

    # Input 2: 3D normal, different bins per dim, custom float weights, density True
    sample = np.random.randn(200, 3).astype(np.float32)
    bins = np.array([4, 5, 6], dtype=np.int32)
    range_val = [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]]
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

    # Input 3: 1D normal representing sample size 50, float32, density False
    sample = np.random.randn(50, 1).astype(np.float32)
    bins = np.array([10], dtype=np.int32)
    range_val = [[-4.0, 4.0]]
    weights = np.random.rand(50).astype(np.float32)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D uniform sample, float64, density True
    sample = np.random.rand(150, 4).astype(np.float64)
    bins = np.array([3, 3, 3, 3], dtype=np.int32)
    range_val = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    weights = np.ones(150, dtype=np.float64)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D sample, float64, 0D tensor for bins, range list, density False
    sample = np.random.randn(80, 2).astype(np.float64)
    bins = np.array(8, dtype=np.int32)
    range_val = [[-3.0, 3.0], [-3.0, 3.0]]
    weights = np.random.rand(80).astype(np.float64)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D sample, asymmetric range, float32, density True
    sample = np.random.randn(120, 2).astype(np.float32)
    bins = np.array([6, 8], dtype=np.int32)
    range_val = [[-2.0, 2.0], [-1.0, 1.0]]
    weights = np.ones(120, dtype=np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D uniform distribution, larger bounds, density False
    sample = np.random.uniform(-5.0, 5.0, (300, 3)).astype(np.float32)
    bins = np.array([5, 5, 5], dtype=np.int32)
    range_val = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    weights = np.random.uniform(0.5, 1.5, 300).astype(np.float32)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 2D sample, float32, density True
    sample = np.random.randn(500, 2).astype(np.float32)
    bins = np.array(12, dtype=np.int32)
    range_val = [[-3.5, 3.5], [-3.5, 3.5]]
    weights = np.random.rand(500).astype(np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D small sample size, float64, density False
    sample = np.random.randn(10, 5).astype(np.float64)
    bins = np.array([2, 2, 2, 2, 2], dtype=np.int32)
    range_val = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
    weights = np.ones(10, dtype=np.float64)
    density = False
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D dense normal sample, high bin count, density True
    sample = np.random.randn(1000, 2).astype(np.float32)
    bins = np.array([20, 20], dtype=np.int32)
    range_val = [[-4.0, 4.0], [-4.0, 4.0]]
    weights = np.random.rand(1000).astype(np.float32)
    density = True
    input_dict = {
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogramdd_2"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_2'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_2'], lib="jax", suffix=2)

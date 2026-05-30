
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogramdd_inputs():
    list_of_inputs = []

    # Input 1: 3D sample with float32, simple integer bins per dimension
    sample = np.random.randn(100, 3).astype(np.float32)
    bins = [5, 6, 7]
    range_val = [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]
    weights = np.random.rand(100).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 2: 2D sample with float64, density enabled
    sample = np.random.rand(50, 2).astype(np.float64)
    bins = [10, 10]
    range_val = [(0.0, 1.0), (0.0, 1.0)]
    weights = np.random.rand(50).astype(np.float64)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 3: 1D sample (N, 1) with integers
    sample = np.random.randint(-10, 11, size=(200, 1)).astype(np.float32)
    bins = [15]
    range_val = [(-10.0, 10.0)]
    weights = np.ones(200, dtype=np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 4: 4D sample with float32, density enabled
    sample = np.random.randn(120, 4).astype(np.float32)
    bins = [4, 4, 4, 4]
    range_val = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    weights = np.random.rand(120).astype(np.float32)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 5: 2D sample, simple integer bins
    sample = np.random.rand(80, 2).astype(np.float32)
    bins = [5, 8]
    range_val = [(0.0, 1.0), (0.0, 1.0)]
    weights = np.random.rand(80).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 6: Small sample size in 3D
    sample = np.random.randn(10, 3).astype(np.float32)
    bins = [3, 3, 3]
    range_val = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    weights = np.random.rand(10).astype(np.float32)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 7: 5D high dimensional sample
    sample = np.random.rand(300, 5).astype(np.float32)
    bins = [5, 5, 5, 5, 5]
    range_val = [(0.0, 1.0)] * 5
    weights = np.random.rand(300).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 8: 2D sample, float64
    sample = np.random.randn(150, 2).astype(np.float64)
    bins = [20, 20]
    range_val = [(-1.0, 1.0), (-1.0, 1.0)]
    weights = np.random.rand(150).astype(np.float64)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 9: Large N, 3D sample with density False
    sample = np.random.randn(1000, 3).astype(np.float32)
    bins = [10, 10, 10]
    range_val = [(-4.0, 4.0)] * 3
    weights = np.random.rand(1000).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 10: 2D sample with asymmetric bins and range
    sample = np.random.randn(500, 2).astype(np.float32)
    bins = [8, 12]
    range_val = [(-1.5, 1.5), (-2.5, 2.5)]
    weights = np.random.rand(500).astype(np.float32)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    return list_of_inputs

generated_inputs["jax.numpy.histogramdd_3"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_3'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_3'], lib="jax", suffix=3)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogramdd_inputs():
    list_of_inputs = []

    # Case 1: D=1, N=100
    sample = np.random.randn(100, 1).astype(np.float32)
    bins = 10
    range_list = [(-3.0, 3.0)]
    weights = np.random.rand(100).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 2: D=2, N=50
    sample = np.random.rand(50, 2).astype(np.float32)
    bins = 5
    range_list = [(0.0, 1.0), (0.0, 1.0)]
    weights = np.ones(50, dtype=np.float32)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 3: D=3, N=200
    sample = np.random.randn(200, 3).astype(np.float64)
    bins = 6
    range_list = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    weights = np.random.randn(200).astype(np.float64)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 4: D=2, N=10
    sample = np.random.uniform(-5.0, 5.0, (10, 2)).astype(np.float32)
    bins = 3
    range_list = [(-5.0, 5.0), (-5.0, 5.0)]
    weights = np.ones(10, dtype=np.float32)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 5: D=4, N=500
    sample = np.random.uniform(0.0, 10.0, (500, 4)).astype(np.float32)
    bins = 4
    range_list = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    weights = np.random.uniform(0.5, 1.5, 500).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 6: D=1, N=1000, float64
    sample = np.random.randn(1000, 1).astype(np.float64)
    bins = 20
    range_list = [(-10.0, 10.0)]
    weights = np.random.rand(1000).astype(np.float64)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 7: D=3, N=20
    sample = np.random.randn(20, 3).astype(np.float32)
    bins = 2
    range_list = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    weights = np.random.rand(20).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 8: D=2, N=300
    sample = np.random.uniform(-5.0, 5.0, (300, 2)).astype(np.float32)
    bins = 8
    range_list = [(-5.0, 0.0), (0.0, 5.0)]
    weights = np.random.rand(300).astype(np.float32)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 9: D=5, N=100
    sample = np.random.randn(100, 5).astype(np.float32)
    bins = 3
    range_list = [(-2.0, 2.0)] * 5
    weights = np.random.randn(100).astype(np.float32)
    density = False
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    # Case 10: D=2, N=150
    sample = np.random.uniform(1.0, 10.0, (150, 2)).astype(np.float32)
    bins = 12
    range_list = [(1.0, 10.0), (1.0, 10.0)]
    weights = np.random.rand(150).astype(np.float32)
    density = True
    list_of_inputs.append({
        "sample": sample,
        "bins": bins,
        "range": range_list,
        "weights": weights,
        "density": density
    })

    return list_of_inputs

generated_inputs["jax.numpy.histogramdd_1"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_1'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_1'], lib="jax", suffix=1)

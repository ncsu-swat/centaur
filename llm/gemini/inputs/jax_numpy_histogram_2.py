
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, 3 bins, density False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bins = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    range_val = (0.0, 6.0)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 2: 2D array, density True
    a = np.random.randn(5, 5).astype(np.float64)
    bins = np.linspace(-3.0, 3.0, 11).astype(np.float64)
    range_val = (-3.0, 3.0)
    weights = np.ones((5, 5), dtype=np.float64)
    density = True
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 3: Negative and positive values, 1D, int32 data
    a = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    bins = np.array([-15, -5, 5, 15], dtype=np.float32)
    range_val = (-15.0, 15.0)
    weights = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    density = False
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 4: Large 1D array
    a = np.random.uniform(10, 20, size=100).astype(np.float32)
    bins = np.linspace(10, 20, 6).astype(np.float32)
    range_val = (10.0, 20.0)
    weights = np.random.uniform(0.1, 1.0, size=100).astype(np.float32)
    density = True
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 5: 3D array of data
    a = np.random.normal(0, 1, size=(2, 3, 4)).astype(np.float32)
    bins = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    range_val = (-2.0, 2.0)
    weights = np.ones((2, 3, 4), dtype=np.float32)
    density = False
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 6: Exponential distribution, density True
    a = np.random.exponential(scale=2.0, size=50).astype(np.float64)
    bins = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float64)
    range_val = (0.0, 10.0)
    weights = np.random.rand(50).astype(np.float64)
    density = True
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 7: Uniform integers with matching size float weights
    a = np.random.randint(0, 50, size=30).astype(np.int64)
    bins = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    range_val = (0, 50)
    weights = np.random.randn(30).astype(np.float32)
    density = False
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 8: Very small range
    a = np.array([0.001, 0.002, 0.0015], dtype=np.float32)
    bins = np.array([0.001, 0.0015, 0.002], dtype=np.float32)
    range_val = (0.001, 0.002)
    weights = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    density = True
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 9: Large integer values
    a = np.array([1000, 2000, 3000, 4000], dtype=np.int32)
    bins = np.array([1000, 2500, 4000], dtype=np.int32)
    range_val = (1000, 4000)
    weights = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    density = False
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

    # Input 10: 1D float64 data with negative limits
    a = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], dtype=np.float64)
    bins = np.array([-6.0, -2.0, 2.0, 6.0], dtype=np.float64)
    range_val = (-6.0, 6.0)
    weights = np.array([0.5, 0.5, 1.0, 1.0, 1.5, 1.5], dtype=np.float64)
    density = True
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density
    })

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

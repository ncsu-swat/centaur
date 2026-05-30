
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_histogram2d_inputs():
    list_of_inputs = []

    # 1. Small size, float32, positive values, density=False
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    y = np.array([4.5, 5.5, 6.5], dtype=np.float32)
    bins = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
    range_val = ((1.0, 4.0), (4.0, 7.0))
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 2. Medium size, float64, positive/negative values, density=True
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float64)
    bins = np.array([[-2.0, 0.0, 2.0, 4.0], [-3.0, 0.0, 3.0, 6.0]], dtype=np.float64)
    range_val = ((-2.0, 4.0), (-3.0, 6.0))
    weights = np.array([0.5, 0.5, 1.5, 2.5], dtype=np.float64)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 3. Large size, float32, random values, density=False
    np.random.seed(42)
    x = np.random.uniform(0, 10, 100).astype(np.float32)
    y = np.random.uniform(0, 10, 100).astype(np.float32)
    bins = np.array([np.linspace(0, 10, 11), np.linspace(0, 10, 11)], dtype=np.float32)
    range_val = ((0.0, 10.0), (0.0, 10.0))
    weights = np.ones(100, dtype=np.float32)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 4. Small size, int32 x/y, float32 weights, density=True
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    bins = np.array([[0, 2, 4, 6], [4, 6, 8, 10]], dtype=np.float32)
    range_val = ((0.0, 6.0), (4.0, 10.0))
    weights = np.array([1.2, 2.3, 3.4, 4.5], dtype=np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 5. Different bin lengths for x and y, density=False
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    bins = np.array([[0.0, 2.0, 4.0, 6.0], [0.0, 3.0, 6.0, 9.0]], dtype=np.float32)
    range_val = ((0.0, 6.0), (0.0, 9.0))
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 6. Negative coordinates, negative weights, density=True
    x = np.array([-10.0, -20.0, -30.0], dtype=np.float32)
    y = np.array([-5.0, -15.0, -25.0], dtype=np.float32)
    bins = np.array([[-40.0, -25.0, -10.0], [-30.0, -15.0, 0.0]], dtype=np.float32)
    range_val = ((-40.0, -10.0), (-30.0, 0.0))
    weights = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 7. Single element arrays
    x = np.array([1.5], dtype=np.float32)
    y = np.array([2.5], dtype=np.float32)
    bins = np.array([[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]], dtype=np.float32)
    range_val = ((0.0, 4.0), (0.0, 4.0))
    weights = np.array([10.0], dtype=np.float32)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 8. Extremely small values, density=True
    x = np.array([1e-5, 2e-5, 3e-5], dtype=np.float32)
    y = np.array([1e-5, 3e-5, 5e-5], dtype=np.float32)
    bins = np.array([[0.0, 2e-5, 4e-5], [0.0, 3e-5, 6e-5]], dtype=np.float32)
    range_val = ((0.0, 4e-5), (0.0, 6e-5))
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 9. Large integers, density=False
    x = np.array([1000, 2000, 3000], dtype=np.int64)
    y = np.array([2000, 4000, 6000], dtype=np.int64)
    bins = np.array([[0, 1500, 3500], [0, 3000, 7000]], dtype=np.float64)
    range_val = ((0.0, 3500.0), (0.0, 7000.0))
    weights = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    density = False
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    # 10. Random normal values, density=True
    x = np.random.normal(0, 1, 50).astype(np.float32)
    y = np.random.normal(0, 1, 50).astype(np.float32)
    bins = np.array([np.linspace(-3, 3, 7), np.linspace(-3, 3, 7)], dtype=np.float32)
    range_val = ((-3.0, 3.0), (-3.0, 3.0))
    weights = np.abs(np.random.normal(1, 0.1, 50)).astype(np.float32)
    density = True
    list_of_inputs.append({
        "x": x, "y": y, "bins": bins, "range": range_val, "weights": weights, "density": density
    })

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_10"] = jax_numpy_histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_10'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_10'], lib="jax", suffix=10)

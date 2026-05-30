
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_histogram2d_inputs():
    list_of_inputs = []

    # 1. Normal distributions, simple integer bins
    x = np.random.randn(100).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    bins = [10, 10]
    range_val = ((-3.0, 3.0), (-3.0, 3.0))
    weights = np.ones(100, dtype=np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 2. Uniform distributions, different bin counts, density True
    x = np.random.uniform(-5, 5, 200).astype(np.float32)
    y = np.random.uniform(-5, 5, 200).astype(np.float32)
    bins = [5, 8]
    range_val = ((-5.0, 5.0), (-5.0, 5.0))
    weights = np.random.rand(200).astype(np.float32)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 3. Simple list of integers for bins
    x = np.random.randn(50).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    bins = [15, 10]
    range_val = ((-2.0, 2.0), (-2.0, 2.0))
    weights = np.ones(50, dtype=np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 4. Linspace values, density True
    x = np.linspace(-10, 10, 300).astype(np.float32)
    y = np.linspace(-10, 10, 300).astype(np.float32)
    bins = [20, 20]
    range_val = ((-10.0, 10.0), (-10.0, 10.0))
    weights = np.random.uniform(0.1, 1.0, 300).astype(np.float32)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 5. Exponential distribution, large inputs, different scale
    x = np.random.exponential(1.0, 150).astype(np.float32)
    y = np.random.exponential(2.0, 150).astype(np.float32)
    bins = [12, 15]
    range_val = ((0.0, 5.0), (0.0, 8.0))
    weights = np.ones(150, dtype=np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 6. Larger dataset with density=True
    x = np.random.normal(0, 1, 1000).astype(np.float32)
    y = np.random.normal(5, 2, 1000).astype(np.float32)
    bins = [50, 50]
    range_val = ((-3.0, 3.0), (-1.0, 11.0))
    weights = np.ones(1000, dtype=np.float32)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 7. Small custom arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    bins = [2, 2]
    range_val = ((1.0, 3.0), (4.0, 6.0))
    weights = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 8. Regular spaced bins, density True
    x = np.random.randn(500).astype(np.float32)
    y = np.random.randn(500).astype(np.float32)
    bins = [30, 30]
    range_val = ((-3.0, 3.0), (-3.0, 3.0))
    weights = np.random.rand(500).astype(np.float32)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 9. float64 inputs
    x = np.random.randint(-10, 10, size=100).astype(np.float64)
    y = np.random.randint(-10, 10, size=100).astype(np.float64)
    bins = [10, 10]
    range_val = ((-10.0, 10.0), (-10.0, 10.0))
    weights = np.ones(100, dtype=np.float64)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # 10. float64 inputs with density=True
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    y = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    bins = [8, 8]
    range_val = ((0.0, 3.0), (0.0, 3.0))
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_8"] = jax_numpy_histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_8'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_8'], lib="jax", suffix=8)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_histogram2d_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.uniform(0.0, 1.0, size=100).astype(np.float32)
    y = np.random.uniform(0.0, 1.0, size=100).astype(np.float32)
    bins = np.linspace(0.0, 1.0, 11).astype(np.float32)
    range_val = ((0.0, 1.0), (0.0, 1.0))
    weights = np.random.uniform(0.5, 1.5, size=100).astype(np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 2
    x = np.random.uniform(-10.0, 10.0, size=50).astype(np.float32)
    y = np.random.uniform(-10.0, 10.0, size=50).astype(np.float32)
    bins = np.linspace(-10.0, 10.0, 6).astype(np.float32)
    range_val = ((-10.0, 10.0), (-10.0, 10.0))
    weights = np.ones(50, dtype=np.float32)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 3
    x = np.random.uniform(-1.0, 1.0, size=200).astype(np.float64)
    y = np.random.uniform(-1.0, 1.0, size=200).astype(np.float64)
    bins = np.linspace(-1.0, 1.0, 21).astype(np.float64)
    range_val = ((-1.0, 1.0), (-1.0, 1.0))
    weights = np.random.uniform(0.0, 1.0, size=200).astype(np.float64)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 4
    x = np.random.randint(-5, 5, size=30).astype(np.int32)
    y = np.random.randint(-5, 5, size=30).astype(np.int32)
    bins = np.array([-5, -2, 0, 2, 5], dtype=np.float32)
    range_val = ((-5.0, 5.0), (-5.0, 5.0))
    weights = np.random.uniform(1.0, 2.0, size=30).astype(np.float32)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 5
    x = np.random.uniform(10.0, 20.0, size=15).astype(np.float32)
    y = np.random.uniform(10.0, 20.0, size=15).astype(np.float32)
    bins = np.linspace(10.0, 20.0, 4).astype(np.float32)
    range_val = ((10.0, 20.0), (10.0, 20.0))
    weights = np.ones(15, dtype=np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 6
    x = np.random.uniform(0.0, 100.0, size=1000).astype(np.float64)
    y = np.random.uniform(0.0, 100.0, size=1000).astype(np.float64)
    bins = np.linspace(0.0, 100.0, 51).astype(np.float64)
    range_val = ((0.0, 100.0), (0.0, 100.0))
    weights = np.random.uniform(0.1, 0.5, size=1000).astype(np.float64)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 7
    x = np.random.uniform(-100.0, 100.0, size=10).astype(np.float32)
    y = np.random.uniform(-100.0, 100.0, size=10).astype(np.float32)
    bins = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    range_val = ((-100.0, 100.0), (-100.0, 100.0))
    weights = np.random.uniform(1.0, 10.0, size=10).astype(np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 8
    x = np.random.uniform(-0.5, 0.5, size=25).astype(np.float32)
    y = np.random.uniform(-0.5, 0.5, size=25).astype(np.float32)
    bins = np.linspace(-0.5, 0.5, 6).astype(np.float32)
    range_val = ((-0.5, 0.5), (-0.5, 0.5))
    weights = np.random.uniform(0.0, 1.0, size=25).astype(np.float32)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 9
    x = np.random.uniform(0.0, 5.0, size=500).astype(np.float32)
    y = np.random.uniform(0.0, 5.0, size=500).astype(np.float32)
    bins = np.linspace(0.0, 5.0, 16).astype(np.float32)
    range_val = ((0.0, 5.0), (0.0, 5.0))
    weights = np.ones(500, dtype=np.float32)
    density = False
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    # Input 10
    x = np.random.uniform(-2.0, 2.0, size=80).astype(np.float64)
    y = np.random.uniform(-2.0, 2.0, size=80).astype(np.float64)
    bins = np.linspace(-2.0, 2.0, 9).astype(np.float64)
    range_val = ((-2.0, 2.0), (-2.0, 2.0))
    weights = np.random.uniform(0.5, 1.5, size=80).astype(np.float64)
    density = True
    list_of_inputs.append({
        'x': x, 'y': y, 'bins': bins, 'range': range_val, 'weights': weights, 'density': density
    })

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_7"] = generate_histogram2d_inputs()

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

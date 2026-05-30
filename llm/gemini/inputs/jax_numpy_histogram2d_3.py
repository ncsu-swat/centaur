
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic floats with equal bins and range, density False
    input_dict = {
        'x': np.random.randn(100).astype(np.float32),
        'y': np.random.randn(100).astype(np.float32),
        'bins': [10, 10],
        'range': [[-3.0, 3.0], [-3.0, 3.0]],
        'weights': np.ones(100).astype(np.float32),
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Linear arrays, density True
    input_dict = {
        'x': np.linspace(-5, 5, 50).astype(np.float32),
        'y': np.linspace(-5, 5, 50).astype(np.float32),
        'bins': [5, 5],
        'range': [[-5.0, 5.0], [-5.0, 5.0]],
        'weights': np.random.rand(50).astype(np.float32),
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Uniform distribution, rectangular bins, different weight scale
    input_dict = {
        'x': np.random.uniform(-10, 10, 200).astype(np.float32),
        'y': np.random.uniform(-5, 15, 200).astype(np.float32),
        'bins': [20, 15],
        'range': [[-10.0, 10.0], [-5.0, 15.0]],
        'weights': np.random.uniform(0.1, 1.0, 200).astype(np.float32),
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 inputs, equal bins
    input_dict = {
        'x': np.random.randn(1000).astype(np.float64),
        'y': np.random.randn(1000).astype(np.float64),
        'bins': [12, 12],
        'range': [[-3.0, 3.0], [-3.0, 3.0]],
        'weights': np.ones(1000).astype(np.float64),
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small array size with float64
    input_dict = {
        'x': np.random.randn(10).astype(np.float64),
        'y': np.random.randn(10).astype(np.float64),
        'bins': [3, 3],
        'range': [[-2.0, 2.0], [-2.0, 2.0]],
        'weights': np.ones(10).astype(np.float64),
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger scale and count, density True
    input_dict = {
        'x': (np.random.rand(10000) * 100).astype(np.float32),
        'y': (np.random.rand(10000) * 100).astype(np.float32),
        'bins': [50, 50],
        'range': [[0.0, 100.0], [0.0, 100.0]],
        'weights': np.random.rand(10000).astype(np.float32),
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer-like floats with large bins
    input_dict = {
        'x': np.random.randint(-100, 100, size=50).astype(np.float32),
        'y': np.random.randint(-100, 100, size=50).astype(np.float32),
        'bins': [8, 12],
        'range': [[-100.0, 100.0], [-100.0, 100.0]],
        'weights': np.random.uniform(0, 10, size=50).astype(np.float32),
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Normal distribution with shifts, density True
    input_dict = {
        'x': np.random.normal(loc=0.0, scale=1.0, size=300).astype(np.float32),
        'y': np.random.normal(loc=5.0, scale=2.0, size=300).astype(np.float32),
        'bins': [15, 15],
        'range': [[-4.0, 4.0], [-1.0, 11.0]],
        'weights': np.ones(300).astype(np.float32),
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small symmetric range with density False
    input_dict = {
        'x': np.random.uniform(-1, 1, 150).astype(np.float32),
        'y': np.random.uniform(-1, 1, 150).astype(np.float32),
        'bins': [6, 6],
        'range': [[-1.0, 1.0], [-1.0, 1.0]],
        'weights': np.random.uniform(1.0, 5.0, 150).astype(np.float32),
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Equal spacing lines, density True
    input_dict = {
        'x': np.linspace(0, 10, 20).astype(np.float32),
        'y': np.linspace(10, 20, 20).astype(np.float32),
        'bins': [4, 4],
        'range': [[0.0, 10.0], [10.0, 20.0]],
        'weights': np.random.rand(20).astype(np.float32),
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_3"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_3'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_3'], lib="jax", suffix=3)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, small positive spacing
    input_dict = {
        'f': np.linspace(-5.0, 5.0, 10, dtype=np.float32),
        'varargs': 1.0,
        'axis': (0,),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32, fractional spacing
    input_dict = {
        'f': np.random.randn(5, 5).astype(np.float32),
        'varargs': 0.5,
        'axis': (0,),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float64, larger spacing, different axis
    input_dict = {
        'f': np.random.randn(4, 4).astype(np.float64),
        'varargs': 2.0,
        'axis': (1,),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float32, multiple axes
    input_dict = {
        'f': np.random.randn(3, 3, 3).astype(np.float32),
        'varargs': 1.5,
        'axis': (0, 1),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array, float64, fine-grained spacing
    input_dict = {
        'f': np.linspace(0.0, 1.0, 20, dtype=np.float64),
        'varargs': 0.1,
        'axis': (0,),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, float32, multiple axes
    input_dict = {
        'f': np.random.randn(6, 6).astype(np.float32),
        'varargs': 1.0,
        'axis': (0, 1),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, float64, specific axis
    input_dict = {
        'f': np.random.randn(4, 4, 4).astype(np.float64),
        'varargs': 0.25,
        'axis': (2,),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, float32, high-dimensional gradient
    input_dict = {
        'f': np.random.randn(2, 3, 4, 5).astype(np.float32),
        'varargs': 1.0,
        'axis': (1, 3),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, float32, large spacing
    input_dict = {
        'f': np.random.randn(3, 10).astype(np.float32),
        'varargs': 3.0,
        'axis': (1,),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, float32, very fine spacing
    input_dict = {
        'f': np.arange(8, dtype=np.float32),
        'varargs': 0.01,
        'axis': (0,),
        'edge_order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_6"] = jax_numpy_gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_6'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_6'], lib="jax", suffix=6)

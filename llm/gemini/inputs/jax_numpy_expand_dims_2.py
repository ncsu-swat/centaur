
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, expanding axis 0
    a = np.random.randn(5).astype(np.float32)
    axis = (0,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 2: 2D float32 array, expanding axis 1
    a = np.random.randn(2, 3).astype(np.float32)
    axis = (1,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 3: 2D int32 array, expanding multiple axes
    a = np.random.randint(0, 10, size=(2, 3)).astype(np.int32)
    axis = (0, 2)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 4: 1D float64 array, expanding negative axis
    a = np.random.randn(4).astype(np.float64)
    axis = (-1,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 5: 3D float32 array, expanding combination of positive and negative axes
    a = np.random.randn(2, 2, 2).astype(np.float32)
    axis = (1, -1)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 6: 0D (scalar) float32 array, expanding axis 0
    a = np.array(42.0).astype(np.float32)
    axis = (0,)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 7: 3D int64 array, expanding multiple positive axes
    a = np.random.randint(0, 2, size=(2, 2, 2)).astype(np.int64)
    axis = (0, 3)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 8: 2D boolean array, expanding multiple leading axes
    a = (np.random.randn(5, 5) > 0).astype(bool)
    axis = (0, 1)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 9: 3D float32 array, expanding inner dimensions
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = (2, -2)
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 10: 1D int16 array, expanding three sequential dimensions
    a = np.random.randint(-10, 10, size=(10,)).astype(np.int16)
    axis = (0, 1, 2)
    list_of_inputs.append({"a": a, "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.expand_dims_2"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expand_dims_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expand_dims_2'.")


check_valid('jax.numpy.expand_dims', generated_inputs['jax.numpy.expand_dims_2'], lib="jax", suffix=2)

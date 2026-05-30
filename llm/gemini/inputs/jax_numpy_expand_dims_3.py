
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=[0]
    a = np.random.randn(5).astype(np.float32)
    axis = [0]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 2: 1D array, int32, axis=[1]
    a = np.arange(5, dtype=np.int32)
    axis = [1]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 3: 2D array, float64, axis=[1]
    a = np.random.randn(3, 4).astype(np.float64)
    axis = [1]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 4: 2D array, bool, axis=[0, 2]
    a = np.random.choice([True, False], size=(3, 3))
    axis = [0, 2]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 5: 3D array, complex64, axis=[-1]
    a = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64)
    axis = [-1]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 6: 1D array, float32, axis=[0, 1, 3]
    a = np.random.randn(3).astype(np.float32)
    axis = [0, 1, 3]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 7: 0D array (scalar), float32, axis=[0]
    a = np.array(4.2, dtype=np.float32)
    axis = [0]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 8: 4D array, float32, axis=[1, 3]
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = [1, 3]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 9: 2D array, int16, axis=[0, 3]
    a = np.random.randint(0, 10, size=(2, 2), dtype=np.int16)
    axis = [0, 3]
    list_of_inputs.append({"a": a, "axis": axis})

    # Input 10: 3D array, float32, axis=[2]
    a = np.random.randn(4, 5, 6).astype(np.float32)
    axis = [2]
    list_of_inputs.append({"a": a, "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.expand_dims_3"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expand_dims_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expand_dims_3'.")


check_valid('jax.numpy.expand_dims', generated_inputs['jax.numpy.expand_dims_3'], lib="jax", suffix=3)

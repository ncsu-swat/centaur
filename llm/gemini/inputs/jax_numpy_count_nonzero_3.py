
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp

# Monkeypatch jax.numpy.count_nonzero to convert list axis to tuple 
# to bypass JAX's limitation with unhashable list static arguments.
orig_count_nonzero = jnp.count_nonzero

def patched_count_nonzero(*args, **kwargs):
    new_args = list(args)
    if 'axis' in kwargs and isinstance(kwargs['axis'], list):
        kwargs['axis'] = tuple(kwargs['axis'])
    elif len(args) > 1 and isinstance(args[1], list):
        new_args[1] = tuple(args[1])
    return orig_count_nonzero(*new_args, **kwargs)

jnp.count_nonzero = patched_count_nonzero

def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: 2D integer array, axis=[0], keepdims=False
    a = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int32)
    input_dict = {"a": a, "axis": [0], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float array, axis=[1], keepdims=True
    a = np.array([[[1.0, 0.0], [0.0, -0.5]], [[0.0, 0.0], [2.3, 0.0]]], dtype=np.float32)
    input_dict = {"a": a, "axis": [1], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D integer array, axis=[0], keepdims=False
    a = np.array([0, -1, 3, 0, 5], dtype=np.int64)
    input_dict = {"a": a, "axis": [0], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D boolean array, axis=[1], keepdims=False
    a = np.array([[True, False, True], [False, False, True]], dtype=bool)
    input_dict = {"a": a, "axis": [1], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D double array with zeros, axis=[0, 2], keepdims=True
    a = np.random.choice([0.0, 1.2, -3.4], size=(3, 4, 2)).astype(np.float64)
    input_dict = {"a": a, "axis": [0, 2], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D int array, axis=[1, 3], keepdims=False
    a = np.random.choice([0, 1], size=(2, 3, 2, 4)).astype(np.int16)
    input_dict = {"a": a, "axis": [1, 3], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float array with negative numbers, axis=[1], keepdims=True
    a = np.array([[-1.0, 0.0, 2.0], [0.0, -3.0, 0.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": [1], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float array with high precision, axis=[0], keepdims=True
    a = np.array([0.0, 0.0001, -0.0002, 0.0], dtype=np.float64)
    input_dict = {"a": a, "axis": [0], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D complex array, axis=[0, 1], keepdims=False
    a = np.array([[[1+1j, 0], [0, 2-2j]], [[0, 0], [3j, 0]]], dtype=np.complex64)
    input_dict = {"a": a, "axis": [0, 1], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int array, axis=[2, 4], keepdims=True
    a = np.random.choice([0, 5], size=(2, 2, 3, 2, 4)).astype(np.int32)
    input_dict = {"a": a, "axis": [2, 4], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.count_nonzero_3"] = count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.count_nonzero_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.count_nonzero_3'.")


check_valid('jax.numpy.count_nonzero', generated_inputs['jax.numpy.count_nonzero_3'], lib="jax", suffix=3)

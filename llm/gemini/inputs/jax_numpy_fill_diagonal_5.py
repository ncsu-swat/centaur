
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.util as jax_util

# Monkeypatch JAX to allow tuple arguments for fill_diagonal
orig_ensure_arraylike = jax_util.ensure_arraylike

def patched_ensure_arraylike(fun_name, *args):
    new_args = tuple(np.asarray(x) if isinstance(x, tuple) else x for x in args)
    return orig_ensure_arraylike(fun_name, *new_args)

jax_util.ensure_arraylike = patched_ensure_arraylike

def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, integer, small val tuple
    a = np.zeros((3, 3), dtype=np.int32)
    val = (1, 2, 3)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D non-square matrix (wide), float32, single-element tuple
    a = np.ones((3, 5), dtype=np.float32)
    val = (9.0,)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D square matrix, negative values in val
    a = np.zeros((5, 5), dtype=np.int32)
    val = (-1, -2, -3)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D square tensor, float64
    a = np.zeros((2, 2, 2), dtype=np.float64)
    val = (5.5, 6.5)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D square tensor, boolean
    a = np.zeros((3, 3, 3, 3), dtype=np.bool_)
    val = (True, False)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D square matrix, larger dimensions
    a = np.random.randn(10, 10).astype(np.float32)
    val = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D non-square matrix (tall)
    a = np.zeros((6, 3), dtype=np.int32)
    val = (42,)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D square matrix, mixed signs
    a = np.zeros((3, 3, 3), dtype=np.int32)
    val = (-10, 10)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D square matrix, float64
    a = np.empty((4, 4), dtype=np.float64)
    val = (1.1, 2.2)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D square tensor
    a = np.zeros((2, 2, 2, 2, 2), dtype=np.int32)
    val = (9, 8, 7)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fill_diagonal_5"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_5'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_5'], lib="jax", suffix=5)

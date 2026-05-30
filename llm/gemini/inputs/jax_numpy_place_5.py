
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.util as jax_util

if not hasattr(jax_util, '_patched_for_place'):
    orig_ensure_arraylike = jax_util.ensure_arraylike
    def patched_ensure_arraylike(fun_name, *args):
        new_args = tuple(np.array(arg) if isinstance(arg, tuple) else arg for arg in args)
        return orig_ensure_arraylike(fun_name, *new_args)
    jax_util.ensure_arraylike = patched_ensure_arraylike
    jax_util._patched_for_place = True

def place_inputs():
    list_of_inputs = []

    # Input 1: 1D int array
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    mask = np.array([True, False, True, False, True], dtype=bool)
    vals = (10, 20)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array
    arr = np.zeros((3, 4), dtype=np.float32)
    mask = np.array([
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False]
    ], dtype=bool)
    vals = (1.5, 2.5, 3.5)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array
    arr = np.ones((2, 2, 2), dtype=np.int32)
    mask = np.ones((2, 2, 2), dtype=bool)
    vals = (-1, -2)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array
    arr = np.linspace(0, 1, 10, dtype=np.float64)
    mask = arr > 0.5
    vals = (9.9, 8.8)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int16 array
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    vals = (100, 200)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D uint8 array
    arr = np.array([10, 20, 30], dtype=np.uint8)
    mask = np.array([True, True, True], dtype=bool)
    vals = (50,)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    arr = np.random.randn(2, 1, 3, 2).astype(np.float32)
    mask = arr > 0
    vals = (0.0, -0.0)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int8 array
    arr = np.zeros((5, 2), dtype=np.int8)
    mask = np.ones((5, 2), dtype=bool)
    vals = (1, 2, 3, 4, 5)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element float32 array
    arr = np.array([1.23], dtype=np.float32)
    mask = np.array([True], dtype=bool)
    vals = (4.56,)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int64 array
    arr = np.zeros((3, 1, 3), dtype=np.int64)
    mask = np.array([
        [[True, False, True]],
        [[False, True, False]],
        [[True, True, True]]
    ], dtype=bool)
    vals = (-10, -20, -30, -40, -50)
    input_dict = {
        "arr": arr,
        "mask": mask,
        "vals": vals,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.place_5"] = place_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.place_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.place_5'.")


check_valid('jax.numpy.place', generated_inputs['jax.numpy.place_5'], lib="jax", suffix=5)

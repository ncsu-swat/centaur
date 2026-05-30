
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.util as jax_util

# Monkeypatch JAX to allow list inputs in place() by converting lists to numpy arrays
_orig_ensure_arraylike = jax_util.ensure_arraylike

def patched_ensure_arraylike(fun_name, *args):
    new_args = tuple(np.array(x) if isinstance(x, list) else x for x in args)
    return _orig_ensure_arraylike(fun_name, *new_args)

jax_util.ensure_arraylike = patched_ensure_arraylike

def place_inputs():
    list_of_inputs = []

    # Input 1: 1D int array, simple mask
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    mask = np.array([True, False, True, False, True], dtype=bool)
    vals = [10, 20, 30]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 2: 2D float array
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    mask = arr > 5
    vals = [-1.0, -2.0, -3.0]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 3: 3D int16 array
    arr = np.zeros((2, 2, 2), dtype=np.int16)
    mask = np.ones((2, 2, 2), dtype=bool)
    vals = [5]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 4: 1D float64 array with negative elements
    arr = np.array([-1.5, -2.5, -3.5, -4.5], dtype=np.float64)
    mask = np.array([False, True, False, True], dtype=bool)
    vals = [9.9, 8.8]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 5: 4D int32 array
    arr = np.ones((2, 1, 3, 2), dtype=np.int32)
    mask = arr == 1
    vals = [100, 200]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 6: 2D boolean array
    arr = np.array([[True, False], [False, True]], dtype=bool)
    mask = np.array([[True, True], [False, False]], dtype=bool)
    vals = [False, True]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 7: 1D uint8 array
    arr = np.array([10, 20, 30], dtype=np.uint8)
    mask = np.array([True, True, True], dtype=bool)
    vals = [255]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 8: 3D float32 array
    arr = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    vals = [0.5]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 9: 2D int64 array
    arr = np.arange(10, dtype=np.int64).reshape(2, 5)
    mask = (arr % 2 == 0)
    vals = [99, 88, 77, 66]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    # Input 10: 1D float32 large array
    arr = np.linspace(0, 10, 100, dtype=np.float32)
    mask = arr < 5.0
    vals = [-10.0]
    inplace = False
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": inplace})

    return list_of_inputs

generated_inputs["jax.numpy.place_4"] = place_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.place_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.place_4'.")


check_valid('jax.numpy.place', generated_inputs['jax.numpy.place_4'], lib="jax", suffix=4)

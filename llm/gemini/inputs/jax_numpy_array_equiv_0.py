
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_equiv_inputs():
    list_of_inputs = []

    # Input 1: Identical 1D arrays of floats
    a1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Broadcastable arrays (2D and 1D) with matching values
    a1 = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
    a2 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Same shape, different values
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([1, 2, 4], dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcastable but not equivalent (2D and 1D)
    a1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a2 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean arrays
    a1 = np.array([True, False, True], dtype=bool)
    a2 = np.array([True, False, True], dtype=bool)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values, identical
    a1 = np.array([-10, -5, 0], dtype=np.int64)
    a2 = np.array([-10, -5, 0], dtype=np.int64)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimensional arrays (3D and 1D broadcastable)
    a1 = np.ones((2, 3, 4), dtype=np.float64)
    a2 = np.array([1.0], dtype=np.float64)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty arrays of same shape
    a1 = np.array([], dtype=np.float32)
    a2 = np.array([], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0-D arrays (scalars)
    a1 = np.array(5, dtype=np.int32)
    a2 = np.array(5, dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Arrays with Inf values
    a1 = np.array([np.inf, -np.inf], dtype=np.float32)
    a2 = np.array([np.inf, -np.inf], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.array_equiv"] = array_equiv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_equiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_equiv'.")


check_valid('jax.numpy.array_equiv', generated_inputs['jax.numpy.array_equiv'], lib="jax", suffix=0)

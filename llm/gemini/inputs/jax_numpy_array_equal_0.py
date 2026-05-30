
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_equal_inputs():
    list_of_inputs = []

    # Input 1: Identical 1D integer arrays
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different 1D integer arrays
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([1, 2, 4], dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Identical 2D float arrays with negative values
    a1 = np.array([[-1.0, 2.5], [3.1, -4.2]], dtype=np.float32)
    a2 = np.array([[-1.0, 2.5], [3.1, -4.2]], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float arrays with NaNs, equal_nan=True
    a1 = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    a2 = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float arrays with NaNs, equal_nan=False
    a1 = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    a2 = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 arrays
    a1 = np.random.randn(2, 3, 4).astype(np.float64)
    a2 = copy.deepcopy(a1)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arrays with different shapes
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([[1, 2, 3]], dtype=np.int32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean arrays
    a1 = np.array([True, False, True], dtype=bool)
    a2 = np.array([True, False, True], dtype=bool)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty 1D arrays
    a1 = np.array([], dtype=np.float32)
    a2 = np.array([], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D arrays with inf and -inf
    a1 = np.array([np.inf, -np.inf, 1.0], dtype=np.float32)
    a2 = np.array([np.inf, -np.inf, 1.0], dtype=np.float32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D random arrays, not equal
    a1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {"a1": a1, "a2": a2, "equal_nan": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.array_equal"] = array_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_equal'.")


check_valid('jax.numpy.array_equal', generated_inputs['jax.numpy.array_equal'], lib="jax", suffix=0)

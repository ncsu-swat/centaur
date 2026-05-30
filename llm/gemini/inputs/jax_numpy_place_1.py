
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_place_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int arrays
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    mask = np.array([True, False, True, False, True], dtype=bool)
    vals = np.array([10, 20], dtype=np.int32)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float arrays with diagonal mask
    arr = np.zeros((3, 3), dtype=np.float32)
    mask = np.array([[True, False, False], [False, True, False], [False, False, True]], dtype=bool)
    vals = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int64 arrays, repeating values
    arr = np.ones((2, 2, 2), dtype=np.int64)
    mask = np.array([[[True, False], [False, True]], [[True, False], [False, True]]], dtype=bool)
    vals = np.array([-1, -2], dtype=np.int64)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 with a single value repeated
    arr = np.arange(10, dtype=np.float64)
    mask = (np.arange(10) % 2 == 0).astype(bool)
    vals = np.array([9.9], dtype=np.float64)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty mask (all False)
    arr = np.array([[-1, -2], [-3, -4]], dtype=np.int16)
    mask = np.array([[False, False], [False, False]], dtype=bool)
    vals = np.array([100], dtype=np.int16)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Full mask (all True) with truncated vals
    arr = np.array([5, 5, 5], dtype=np.uint8)
    mask = np.array([True, True, True], dtype=bool)
    vals = np.array([1, 2, 3, 4], dtype=np.uint8)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, full mask, single negative value
    arr = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    mask = np.ones((3, 4), dtype=bool)
    vals = np.array([-0.5], dtype=np.float32)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, single True in mask
    arr = np.array([10, 20, 30], dtype=np.int64)
    mask = np.array([False, True, False], dtype=bool)
    vals = np.array([99], dtype=np.int64)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 3D arrays with random mask
    arr = np.random.randint(-100, 100, size=(4, 4, 4), dtype=np.int32)
    mask = (np.random.rand(4, 4, 4) > 0.5).astype(bool)
    vals = np.array([0, -1, 1], dtype=np.int32)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D double arrays with randomly generated elements
    arr = np.random.randn(5, 5).astype(np.float64)
    mask = (np.random.randn(5, 5) > 0).astype(bool)
    vals = np.random.randn(10).astype(np.float64)
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.place_1"] = jax_numpy_place_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.place_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.place_1'.")


check_valid('jax.numpy.place', generated_inputs['jax.numpy.place_1'], lib="jax", suffix=1)

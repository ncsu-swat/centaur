
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D (C-order)
    a = np.arange(6, dtype=np.int32)
    shape = [2, 3]
    order = "C"
    copy_val = True
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D to 1D (F-order)
    a = np.random.randn(3, 4).astype(np.float32)
    shape = [12]
    order = "F"
    copy_val = False
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Auto-inference of dimension using -1
    a = np.random.randn(2, 3, 4).astype(np.float32)
    shape = [-1, 6]
    order = "C"
    copy_val = True
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: High-dimensional array reshape
    a = np.random.randint(0, 10, size=(1, 2, 3, 4)).astype(np.int64)
    shape = [4, 3, 2, 1]
    order = "F"
    copy_val = False
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 dtype, flat shape
    a = np.random.randn(100).astype(np.float64)
    shape = [10, 10]
    order = "C"
    copy_val = True
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D reshape with auto-inference
    a = np.random.randn(2, 5, 2).astype(np.float32)
    shape = [2, -1, 5]
    order = "C"
    copy_val = False
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small integer array, preserving size
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = [4]
    order = "F"
    copy_val = True
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Flattening using -1
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    shape = [-1]
    order = "C"
    copy_val = True
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large float32 array
    a = np.ones((100, 10), dtype=np.float32)
    shape = [10, 10, 10]
    order = "C"
    copy_val = False
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean array
    a = np.random.choice([True, False], size=(4, 4))
    shape = [2, 8]
    order = "F"
    copy_val = True
    input_dict = {"a": a, "shape": shape, "order": order, "copy": copy_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.reshape_3"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reshape_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reshape_3'.")


check_valid('jax.numpy.reshape', generated_inputs['jax.numpy.reshape_3'], lib="jax", suffix=3)

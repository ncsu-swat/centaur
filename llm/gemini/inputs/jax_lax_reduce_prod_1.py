
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def reduce_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce along axis 0
    operand = np.array([1.5, -2.0, 3.0, 4.5], dtype=np.float32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 2: 2D int32 array with negative values, reduce along axis 1
    operand = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 3: 2D float64 array, reduce along axis 0
    operand = np.array([[0.5, 1.2], [-3.4, 2.1], [1.0, -0.5]], dtype=np.float64)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 4: 3D float32 array, reduce along two axes
    operand = np.random.uniform(0.5, 1.5, size=(2, 3, 4)).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 5: 3D int64 array, empty axes (identity reduction)
    operand = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 6: 4D float32 array, reduce over three axes
    operand = np.random.uniform(-2.0, 2.0, size=(2, 2, 3, 2)).astype(np.float32)
    axes = [1, 2, 3]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 7: 1D int32 array with negative values, empty axes
    operand = np.array([-3, 5, -7, 11], dtype=np.int32)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 8: 2D float32 array, reduce along axis 1
    operand = np.array([[1.0, -1.0, 2.0], [3.0, -2.0, 0.5]], dtype=np.float32)
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 9: 5D float32 array, reduce over non-consecutive axes
    operand = np.ones((2, 3, 2, 2, 3), dtype=np.float32) * 1.1
    axes = [0, 4]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 10: 3D float32 array, reduce all axes
    operand = np.array([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 1.0], [2.0, -2.0]]], dtype=np.float32)
    axes = [0, 1, 2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 11: 2D int64 array, reduce along axis 0
    operand = np.array([[2, 3, 4], [5, 1, 2]], dtype=np.int64)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    return list_of_inputs

generated_inputs["jax.lax.reduce_prod_1"] = reduce_prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_prod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_prod_1'.")


check_valid('jax.lax.reduce_prod', generated_inputs['jax.lax.reduce_prod_1'], lib="jax", suffix=1)

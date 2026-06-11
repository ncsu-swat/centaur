
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reduce_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce along axis 0
    operand = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    axes = [0]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 2: 2D int32 array, reduce along axis 0
    operand = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axes = [0]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 3: 2D float64 array, reduce along axis 1
    operand = np.random.randn(3, 4).astype(np.float64)
    axes = [1]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 4: 3D float32 array, reduce along multiple axes [0, 2]
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 5: 3D int16 array, empty axes list (no reduction, returns copy)
    operand = np.arange(1, 9, dtype=np.int16).reshape(2, 2, 2)
    axes = []
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 6: 4D float32 array, reduce along axis 3
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axes = [3]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 7: 2D int64 array with negative elements, reduce along axis 0
    operand = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int64)
    axes = [0]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 8: 1D float32 array with a single element, reduce along axis 0
    operand = np.array([42.0], dtype=np.float32)
    axes = [0]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 9: 5D float32 array, reduce along multiple axes [1, 3, 4]
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axes = [1, 3, 4]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 10: 2D float32 array, reduce all axes [0, 1]
    operand = np.array([[0.5, 1.5], [2.0, 2.5]], dtype=np.float32)
    axes = [0, 1]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

    # Input 11: 3D int32 array, reduce along axis 1
    operand = np.random.randint(-5, 5, size=(3, 2, 4)).astype(np.int32)
    axes = [1]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "axes": copy.deepcopy(axes)})

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

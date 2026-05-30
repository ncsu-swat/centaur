
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def block_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D concatenation (homogeneous shape)
    arrays = [np.array([1, 2]), np.array([3, 4])]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 2: 2D Block of square matrices (homogeneous shape)
    arrays = [
        [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)],
        [np.full((2, 2), 2.0, dtype=np.float32), np.full((2, 2), 3.0, dtype=np.float32)]
    ]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 3: 2D block with 3x3 matrices (homogeneous shape)
    arrays = [
        [np.zeros((3, 3), dtype=np.float32), np.ones((3, 3), dtype=np.float32)],
        [np.ones((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32)]
    ]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 4: Nested list with int32 (homogeneous shape)
    x = np.arange(4).reshape((2, 2)).astype(np.int32)
    arrays = [[x, x], [x, x]]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 5: float64 elements in a 2D block structure (homogeneous shape)
    arrays = [[np.array([[1.0, 2.0]], dtype=np.float64), np.array([[3.0, 4.0]], dtype=np.float64)]]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 6: boolean blocks (homogeneous shape)
    arrays = [[np.array([[True]], dtype=bool)], [np.array([[False]], dtype=bool)]]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 7: 1D int64 blocks (homogeneous shape)
    arrays = [np.array([-1, -2], dtype=np.int64), np.array([-4, -5], dtype=np.int64)]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 8: Complex blocks (homogeneous shape)
    arrays = [[np.array([[1+1j]], dtype=np.complex64), np.array([[2-2j]], dtype=np.complex64)]]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 9: Multi-nested block with uint8 (homogeneous shape)
    arrays = [[np.array([[10]], dtype=np.uint8)], [np.array([[20]], dtype=np.uint8)]]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    # Input 10: Larger inputs with random values (homogeneous shape)
    a = np.random.randn(5, 5).astype(np.float32)
    arrays = [[a, a], [a, a]]
    list_of_inputs.append({"arrays": copy.deepcopy(arrays)})

    return list_of_inputs

generated_inputs["jax.numpy.block_3"] = block_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.block_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.block_3'.")


check_valid('jax.numpy.block', generated_inputs['jax.numpy.block_3'], lib="jax", suffix=3)

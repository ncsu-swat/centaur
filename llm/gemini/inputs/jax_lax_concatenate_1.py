
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array (unpacked to 1D arrays) along dimension 0
    op1 = np.array([[1.0, -2.0, 3.5], [4.0, 5.5, -6.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"operands": op1, "dimension": 0}))

    # Input 2: 2D int32 array along dimension 0
    op2 = np.array(
        [[1, 2, 3, 4], [-5, -6, 7, 8], [9, 10, -11, 12]], dtype=np.int32
    )
    list_of_inputs.append(copy.deepcopy({"operands": op2, "dimension": 0}))

    # Input 3: 3D float32 array along dimension 0
    op3 = np.random.randn(2, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"operands": op3, "dimension": 0}))

    # Input 4: 3D float32 array along dimension 1
    op4 = np.random.randn(3, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"operands": op4, "dimension": 1}))

    # Input 5: 3D int64 array along dimension 0
    op5 = np.random.randint(-100, 100, size=(4, 2, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"operands": op5, "dimension": 0}))

    # Input 6: 4D float64 array along dimension 2
    op6 = np.random.randn(2, 3, 2, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"operands": op6, "dimension": 2}))

    # Input 7: 4D float32 array along dimension 1
    op7 = np.random.randn(3, 2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"operands": op7, "dimension": 1}))

    # Input 8: 2D bool array along dimension 0
    op8 = np.array(
        [[True, False, True, False, True], [False, True, False, True, False]],
        dtype=np.bool_,
    )
    list_of_inputs.append(copy.deepcopy({"operands": op8, "dimension": 0}))

    # Input 9: 3D complex64 array along dimension 1
    op9 = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"operands": op9, "dimension": 1}))

    # Input 10: 5D int32 array along dimension 3
    op10 = np.random.randint(-10, 10, size=(2, 2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"operands": op10, "dimension": 3}))

    # Input 11: 4D float16 array along dimension 0
    op11 = np.random.randn(2, 3, 4, 5).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"operands": op11, "dimension": 0}))

    return list_of_inputs


generated_inputs["jax.lax.concatenate_1"] = concatenate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.concatenate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.concatenate_1'.")


check_valid('jax.lax.concatenate', generated_inputs['jax.lax.concatenate_1'], lib="jax", suffix=1)

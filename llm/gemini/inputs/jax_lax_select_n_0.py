
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def select_n_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, bool "which" (all False), 1 case, float32
    which = np.zeros(5, dtype=bool)
    cases = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, int32 "which" (all 0), 1 case, float32
    which = np.zeros((3, 4), dtype=np.int32)
    cases = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar "which" (int32 = 0), 1 case, 3D array, int32
    which = np.array(0, dtype=np.int32)
    cases = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays, int32 "which" (all 0), 1 case, float64
    which = np.zeros(10, dtype=np.int32)
    cases = np.random.randn(10).astype(np.float64)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar "which" (bool = False), 1 case, scalar, float32
    which = np.array(False, dtype=bool)
    cases = np.array(1.5, dtype=np.float32)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays, int32 "which" (all 0), 1 case, int64
    which = np.zeros((2, 2), dtype=np.int32)
    cases = np.random.randint(-100, 100, size=(2, 2)).astype(np.int64)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D arrays, bool "which" (all False), 1 case, float16
    which = np.zeros((2, 2, 2, 2), dtype=bool)
    cases = np.random.randn(2, 2, 2, 2).astype(np.float16)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D arrays, int32 "which" (all 0), 1 case, boolean
    which = np.zeros(6, dtype=np.int32)
    cases = np.random.choice([True, False], size=(6,)).astype(bool)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays, int32 "which" (all 0), 1 case, float32, with negative values in cases
    which = np.zeros((3, 1, 3), dtype=np.int32)
    cases = (np.random.randn(3, 1, 3) * 10.0).astype(np.float32)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays, int8 "which" (all 0), 1 case, complex64
    which = np.zeros((2, 3), dtype=np.int8)
    cases = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    input_dict = {"which": which, "cases": cases}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.select_n"] = select_n_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.select_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.select_n'.")


check_valid('jax.lax.select_n', generated_inputs['jax.lax.select_n'], lib="jax", suffix=0)

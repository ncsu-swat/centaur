
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 array, small sizes
    operand = np.array(3.14, dtype=np.float32)
    sizes = [2, 3]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 2: 1D int32 array, single-element sizes
    operand = np.array([1, 2, 3, 4], dtype=np.int32)
    sizes = [5]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 3: 2D float32 array, single-element sizes
    operand = np.random.randn(3, 4).astype(np.float32)
    sizes = [2]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 4: 2D float64 array, multi-element sizes
    operand = np.random.randn(5, 5).astype(np.float64)
    sizes = [10, 10]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 5: 3D complex64 array, multi-element sizes
    operand = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    sizes = [1, 5]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 6: 1D bool array, multi-element sizes
    operand = np.array([True, False, True], dtype=np.bool_)
    sizes = [2, 2, 2]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 7: 4D float32 array, single-element sizes
    operand = np.random.randn(2, 1, 3, 1).astype(np.float32)
    sizes = [3]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 8: 0D int16 array, single-element sizes
    operand = np.array(-42, dtype=np.int16)
    sizes = [4]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 9: 2D uint8 array, multi-element sizes
    operand = np.random.randint(0, 255, size=(10, 10)).astype(np.uint8)
    sizes = [2, 3, 4]
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    # Input 10: 1D float32 array, empty sizes (no leading dimensions added)
    operand = np.random.randn(100).astype(np.float32)
    sizes = []
    list_of_inputs.append({"operand": copy.deepcopy(operand), "sizes": copy.deepcopy(sizes)})

    return list_of_inputs

generated_inputs["jax.lax.broadcast_2"] = broadcast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_2'.")


check_valid('jax.lax.broadcast', generated_inputs['jax.lax.broadcast_2'], lib="jax", suffix=2)

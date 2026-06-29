
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, simple split
    operand = np.arange(10).astype(np.float32)
    sizes = [3, 7]
    axis = 0
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 2: 2D array, split along axis 0 into equal parts
    operand = np.random.randn(6, 4).astype(np.float32)
    sizes = [2, 2, 2]
    axis = 0
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 3: 2D array, split along axis 1 into unequal parts
    operand = np.random.randn(6, 4).astype(np.float32)
    sizes = [1, 3]
    axis = 1
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 4: 3D float64 array, split along axis 0
    operand = np.random.randn(2, 4, 6).astype(np.float64)
    sizes = [1, 1]
    axis = 0
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 5: 3D int32 array, split along axis 1
    operand = np.random.randint(0, 10, size=(2, 4, 6)).astype(np.int32)
    sizes = [2, 2]
    axis = 1
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 6: 3D array, split along axis 2 with multiple chunks
    operand = np.random.randn(2, 4, 6).astype(np.float32)
    sizes = [1, 2, 3]
    axis = 2
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 7: 2D array, split along negative axis
    operand = np.random.randn(3, 5).astype(np.float32)
    sizes = [2, 3]
    axis = -1
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 8: 1D array, single split returning the whole array
    operand = np.arange(5).astype(np.int64)
    sizes = [5]
    axis = 0
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 9: 4D float16 array, split along axis 3
    operand = np.random.randn(2, 3, 4, 10).astype(np.float16)
    sizes = [3, 3, 4]
    axis = 3
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    # Input 10: Boolean array split
    operand = (np.random.randn(4, 4) > 0).astype(np.bool_)
    sizes = [2, 2]
    axis = 0
    list_of_inputs.append({"operand": operand, "sizes": sizes, "axis": axis})

    return list_of_inputs

generated_inputs["jax.lax.split_1"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.split_1'.")


check_valid('jax.lax.split', generated_inputs['jax.lax.split_1'], lib="jax", suffix=1)

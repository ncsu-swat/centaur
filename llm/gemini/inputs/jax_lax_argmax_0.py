
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, int32 index
    operand = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    input_dict = {"operand": operand, "axis": 0, "index_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis 1, int64 index
    operand = np.random.randn(4, 6).astype(np.float32)
    input_dict = {"operand": operand, "axis": 1, "index_dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 0, int32 index
    operand = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"operand": operand, "axis": 0, "index_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array, axis 2, int32 index
    operand = np.random.randint(-50, 50, size=(2, 3, 4), dtype=np.int32)
    input_dict = {"operand": operand, "axis": 2, "index_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, axis 1, int64 index
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"operand": operand, "axis": 1, "index_dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array with negative values, axis 0, int32 index
    operand = np.array([-10, -3, -20, -1, -5], dtype=np.int64)
    input_dict = {"operand": operand, "axis": 0, "index_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, axis 3, int64 index
    operand = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {"operand": operand, "axis": 3, "index_dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float16 array, axis 0, int32 index
    operand = np.random.randn(3, 8).astype(np.float16)
    input_dict = {"operand": operand, "axis": 0, "index_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D uint8 array, axis 1, int32 index
    operand = np.random.randint(0, 255, size=(4, 7), dtype=np.uint8)
    input_dict = {"operand": operand, "axis": 1, "index_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array containing inf and -inf, axis 0, int64 index
    operand = np.array([-np.inf, 0.0, 1.0, np.inf, 2.0], dtype=np.float32)
    input_dict = {"operand": operand, "axis": 0, "index_dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D int16 array, axis 0, int32 index
    operand = np.random.randint(-100, 100, size=(3, 2, 2, 2), dtype=np.int16)
    input_dict = {"operand": operand, "axis": 0, "index_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.argmax"] = argmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.argmax'.")


check_valid('jax.lax.argmax', generated_inputs['jax.lax.argmax'], lib="jax", suffix=0)

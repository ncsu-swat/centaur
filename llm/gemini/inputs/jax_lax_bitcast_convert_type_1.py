
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


def bitcast_convert_type_inputs():
    list_of_inputs = []

    # Input 1: Same itemsize (float32 -> int32), 2D array with negative values
    operand = np.array([[-1.5, 2.0, 3.5], [0.0, -3.25, 4.0]], dtype=np.float32)
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 2: Same itemsize (int32 -> float32), 1D array
    operand = np.array([-10, 0, 10, 1065353216], dtype=np.int32)
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 3: Same itemsize (int32 -> float32), 3D array
    operand = np.arange(16, dtype=np.int32).reshape((2, 2, 4)) - 8
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 4: Same itemsize (float32 -> int32), 1D array
    operand = np.array([0.0, -1.0, 1.0], dtype=np.float32)
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 5: Same itemsize (int8 -> int8), 2D array
    operand = np.array([[-128, -1, 5], [0, 127, -50]], dtype=np.int8)
    new_dtype = np.dtype(np.int8)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 6: Smaller itemsize (int32 -> int8), 1D array
    operand = np.array([1000, -2000, 3000], dtype=np.int32)
    new_dtype = np.dtype(np.int8)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 7: Smaller itemsize (int32 -> int8), 2D array
    operand = np.arange(10, dtype=np.int32).reshape((2, 5)) - 5
    new_dtype = np.dtype(np.int8)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 8: Smaller itemsize (float32 -> int8), 3D array
    operand = np.random.randn(1, 3, 2).astype(np.float32)
    new_dtype = np.dtype(np.int8)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 9: Smaller itemsize (float32 -> int8), 1D array
    operand = np.array([1.0, -2.5, 0.0, 100.0], dtype=np.float32)
    new_dtype = np.dtype(np.int8)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 10: Larger itemsize (int8 -> int32), last dim == 4 (4 * 1 == 4)
    operand = np.arange(12, dtype=np.int8).reshape((3, 4))
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 11: Larger itemsize (int8 -> float32), 1D array, last dim == 4 (4 * 1 == 4)
    operand = np.array([0, 0, -128, 63], dtype=np.int8)
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 12: Larger itemsize (int8 -> int32), 3D array, last dim == 4 (4 * 1 == 4)
    operand = np.arange(24, dtype=np.int8).reshape((2, 3, 4)) - 12
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    return list_of_inputs


generated_inputs["jax.lax.bitcast_convert_type_1"] = bitcast_convert_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitcast_convert_type_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitcast_convert_type_1'.")


check_valid('jax.lax.bitcast_convert_type', generated_inputs['jax.lax.bitcast_convert_type_1'], lib="jax", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argmin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, int32 index_dtype
    operand = np.array([3.5, 1.2, -4.8, 0.0, 7.3], dtype=np.float32)
    axis = 0
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 2: 2D float32 array, axis 0, int32 index_dtype
    operand = np.random.randn(5, 4).astype(np.float32)
    axis = 0
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 3: 2D int32 array with negative values, axis 1, int32 index_dtype
    operand = np.array([[-10, 20, 5], [3, -15, 0]], dtype=np.int32)
    axis = 1
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 4: 3D float32 array, axis 2, int32 index_dtype
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 5: 3D int32 array, axis 1, int32 index_dtype
    operand = np.random.randint(-100, 100, size=(4, 6, 2), dtype=np.int32)
    axis = 1
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 6: 4D float32 array, axis 2, int32 index_dtype
    operand = np.random.randn(2, 2, 3, 2).astype(np.float32)
    axis = 2
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 7: 1D int32 array, axis 0, int32 index_dtype
    operand = np.array([100, 42, 5, 999], dtype=np.int32)
    axis = 0
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 8: 2D float32 array, axis 0, int32 index_dtype
    operand = np.array(
        [[1.0, 2.0], [0.5, -0.5], [-2.0, 3.0]], dtype=np.float32
    )
    axis = 0
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 9: 5D int32 array, axis 3, int32 index_dtype
    operand = np.random.randint(-50, 50, size=(2, 1, 3, 4, 2), dtype=np.int32)
    axis = 3
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    # Input 10: 2D float32 array with a single row, axis 1, int32 index_dtype
    operand = np.array([[10.0, -1.0, 3.0, 0.0]], dtype=np.float32)
    axis = 1
    index_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "axis": axis, "index_dtype": index_dtype}
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.argmin"] = argmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.argmin'.")


check_valid('jax.lax.argmin', generated_inputs['jax.lax.argmin'], lib="jax", suffix=0)

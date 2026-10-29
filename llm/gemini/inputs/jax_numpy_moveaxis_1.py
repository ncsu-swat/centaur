
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def moveaxis_inputs():
    list_of_inputs = []

    # Input 1: 2D array, move first axis to second axis
    a = np.ones((2, 3), dtype=np.float32)
    source = 0
    destination = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 2: 2D array, move second axis to first axis using negative source
    a = np.zeros((4, 5), dtype=np.int32)
    source = -1
    destination = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 3: 3D array, move axis 0 to the end using negative destination
    a = np.random.randn(2, 3, 4).astype(np.float64)
    source = 0
    destination = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 4: 3D array, move middle axis to beginning
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    source = 1
    destination = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 5: 1D array, moving axis 0 to 0 (identity)
    a = np.array([1, 2, 3, 4], dtype=np.int8)
    source = 0
    destination = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 6: 4D array, move axis using both negative indices
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    source = -2
    destination = -4
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 7: 4D boolean array, move axis 3 to axis 1
    a = np.ones((2, 2, 2, 2), dtype=np.bool_)
    source = 3
    destination = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 8: 5D complex array, move axis 4 to axis 2
    a = np.zeros((1, 2, 3, 4, 5), dtype=np.complex64)
    source = 4
    destination = 2
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 9: 3D array, move last axis to middle
    a = np.random.uniform(-1, 1, size=(6, 7, 8)).astype(np.float32)
    source = 2
    destination = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 10: 5D array, move first axis to last with negative source and destination
    a = np.arange(32, dtype=np.uint8).reshape((2, 2, 2, 2, 2))
    source = -5
    destination = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 11: 1D array with negative indices
    a = np.array([10.0, 20.0, 30.0], dtype=np.float16)
    source = -1
    destination = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    return list_of_inputs


generated_inputs["jax.numpy.moveaxis_1"] = moveaxis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.moveaxis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.moveaxis_1'.")


check_valid('jax.numpy.moveaxis', generated_inputs['jax.numpy.moveaxis_1'], lib="jax", suffix=1)

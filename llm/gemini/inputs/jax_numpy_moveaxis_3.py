
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def moveaxis_inputs():
    list_of_inputs = []

    # Input 1: 1D array, moving single axis
    a = np.ones((5,), dtype=np.float32)
    source = (0,)
    destination = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 2: 2D array, moving row to column
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    source = (0,)
    destination = (1,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 3: 2D array, using negative indices
    a = np.random.randn(4, 5).astype(np.float64)
    source = (-1,)
    destination = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 4: 3D array, moving two axes simultaneously
    a = np.zeros((2, 3, 4), dtype=np.bool_)
    source = (0, 1)
    destination = (1, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 5: 3D array, full axis permutation using positive and negative indices
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = (0, 1, 2)
    destination = (-1, 0, 1)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 6: 4D array, moving axes to the end
    a = np.ones((2, 3, 4, 5), dtype=np.complex64)
    source = (0, 1)
    destination = (-2, -1)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 7: 4D array, swapping inner axes
    a = np.random.randint(0, 10, size=(3, 4, 5, 6), dtype=np.int64)
    source = (1, 2)
    destination = (2, 1)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 8: 5D array, float16 dtype
    a = np.ones((1, 2, 3, 4, 5), dtype=np.float16)
    source = (0, 4)
    destination = (4, 0)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 9: 5D array, moving negative indices to positive indices
    a = np.zeros((2, 2, 3, 3, 4), dtype=np.uint8)
    source = (-1, -3)
    destination = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 10: 6D array, moving multiple non-consecutive axes
    a = np.random.randn(2, 3, 2, 4, 2, 5).astype(np.float32)
    source = (1, 3, 5)
    destination = (0, 2, 4)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    return list_of_inputs


generated_inputs["jax.numpy.moveaxis_3"] = moveaxis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.moveaxis_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.moveaxis_3'.")


check_valid('jax.numpy.moveaxis', generated_inputs['jax.numpy.moveaxis_3'], lib="jax", suffix=3)

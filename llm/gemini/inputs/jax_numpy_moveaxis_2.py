
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def moveaxis_inputs():
    list_of_inputs = []

    # Input 1: 2D array, moving axis 0 to axis 1
    a = np.random.randn(3, 4).astype(np.float32)
    source = [0]
    destination = [1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 2: 3D array, moving axis 0 to axis 2
    a = np.random.randn(2, 3, 4).astype(np.float64)
    source = [0]
    destination = [2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 3: 3D array, using negative indexing for source
    a = np.random.randint(0, 10, size=(2, 3, 4), dtype=np.int32)
    source = [-1]
    destination = [0]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 4: 4D array, moving multiple axes with negative destinations
    a = np.random.randint(-5, 5, size=(2, 3, 4, 5), dtype=np.int64)
    source = [0, 1]
    destination = [-1, -2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 5: 4D array, moving multiple negative axes to positive destinations
    a = np.random.randn(3, 4, 5, 6).astype(np.float32)
    source = [-2, -1]
    destination = [0, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 6: 1D array, moving axis 0 to axis 0
    a = np.arange(10, dtype=np.uint8)
    source = [0]
    destination = [0]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 7: 5D array, swapping positions of two axes
    a = np.random.randn(2, 2, 3, 4, 2).astype(np.float64)
    source = [1, 3]
    destination = [3, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 8: 3D boolean array, moving two axes
    a = np.random.choice([True, False], size=(4, 5, 6))
    source = [0, 2]
    destination = [1, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 9: 4D complex array, moving three axes simultaneously
    a = (np.random.randn(2, 3, 2, 3) + 1j * np.random.randn(2, 3, 2, 3)).astype(
        np.complex64
    )
    source = [1, 2, 3]
    destination = [0, 1, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 10: 2D array, negative source and negative destination
    a = np.zeros((5, 7), dtype=np.int16)
    source = [-1]
    destination = [-2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    # Input 11: 3D array, full cyclic permutation of axes
    a = np.ones((2, 3, 4), dtype=np.float32)
    source = [0, 1, 2]
    destination = [2, 0, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "source": source, "destination": destination})
    )

    return list_of_inputs


generated_inputs["jax.numpy.moveaxis_2"] = moveaxis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.moveaxis_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.moveaxis_2'.")


check_valid('jax.numpy.moveaxis', generated_inputs['jax.numpy.moveaxis_2'], lib="jax", suffix=2)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def moveaxis_inputs():
    list_of_inputs = []

    # Input 1: 3D array, move one axis
    a = np.random.randn(2, 3, 4).astype(np.float32)
    source = (0,)
    destination = (2,)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 2: 3D array, move multiple axes
    a = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int32)
    source = (0, 1)
    destination = (2, 0)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 3: 4D array, negative index
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    source = (-1,)
    destination = (1,)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 4: 4D array, boolean, multiple negative indices
    a = (np.random.randn(2, 3, 4, 5) > 0).astype(np.bool_)
    source = (0, -1)
    destination = (-1, -2)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 5: 2D array, complex values
    a = (np.random.randn(5, 10) + 1j * np.random.randn(5, 10)).astype(np.complex64)
    source = (1, 0)
    destination = (0, 1)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 6: 5D array, move non-contiguous axes
    a = np.random.randn(2, 2, 3, 3, 4).astype(np.float32)
    source = (1, 3)
    destination = (0, 4)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 7: 3D array, int64
    a = np.random.randint(-100, 100, size=(10, 20, 30)).astype(np.int64)
    source = (2,)
    destination = (0,)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 8: 4D array, all negative indices
    a = np.random.randn(3, 5, 7, 9).astype(np.float32)
    source = (-3, -2)
    destination = (-1, -4)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 9: 2D array, single element axes
    a = np.random.randn(1, 5).astype(np.float32)
    source = (0,)
    destination = (1,)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

    # Input 10: 5D array, move three axes
    a = np.random.randint(0, 5, size=(2, 3, 4, 5, 6)).astype(np.int16)
    source = (0, 2, 4)
    destination = (4, 0, 2)
    list_of_inputs.append({"a": a, "source": source, "destination": destination})

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

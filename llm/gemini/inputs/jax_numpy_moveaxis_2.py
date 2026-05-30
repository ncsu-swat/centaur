
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def moveaxis_inputs():
    list_of_inputs = []

    # Input 1: Move single axis in 3D float32 array
    a = np.random.randn(2, 3, 4).astype(np.float32)
    source = [0]
    destination = [2]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Move multiple axes in 4D float32 array
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    source = [0, 1]
    destination = [-1, -2]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Move single axis in 2D int32 array
    a = np.random.randint(0, 10, size=(5, 10)).astype(np.int32)
    source = [1]
    destination = [0]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Move multiple axes in 5D float64 array using positive indices
    a = np.random.randn(2, 3, 4, 5, 6).astype(np.float64)
    source = [1, 2]
    destination = [3, 4]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Move single negative axis in 3D int64 array
    a = np.random.randint(-100, 100, size=(4, 5, 6)).astype(np.int64)
    source = [-1]
    destination = [0]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Move axis in 1D float32 array (identity-like move)
    a = np.random.randn(10).astype(np.float32)
    source = [0]
    destination = [0]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Move multiple non-adjacent axes in 4D float32 array
    a = np.random.randn(3, 4, 5, 6).astype(np.float32)
    source = [1, 3]
    destination = [0, 2]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Move axes in 5D boolean array
    a = np.random.choice([True, False], size=(2, 2, 3, 3, 4))
    source = [0, -1]
    destination = [-2, 1]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Move axes in 3D complex64 array
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    source = [2, 0]
    destination = [1, 2]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Move axes in a 6D float32 array with mixed negative and positive indices
    a = np.random.randn(2, 2, 3, 3, 4, 4).astype(np.float32)
    source = [-3, -1]
    destination = [0, 1]
    input_dict = {"a": a, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

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

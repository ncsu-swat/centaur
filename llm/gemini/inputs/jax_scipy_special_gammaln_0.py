
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gammaln_inputs():
    list_of_inputs = []

    # Input 1: 1D positive float32 array
    x = np.array([1.0, 2.0, 3.5, 4.2, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array with negative non-integer values (float32)
    x = np.array([[-0.5, -1.5], [-2.3, -3.7]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar float64 tensor
    x = np.array(5.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D integer array
    x = np.array([1, 2, 3, 5, 8, 13], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 array with positive random values
    x = np.random.uniform(0.1, 20.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D float64 array with mixed positive and negative non-integers
    x = np.array([[-3.2, 0.5, 1.2], [-0.1, 4.8, -1.9]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D float32 array
    x = np.random.uniform(1.0, 50.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D float64 array with half-integers
    x = np.arange(0.5, 10.5, 1.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D float32 array with large values
    x = np.array([50.0, 100.0, 500.0, 1000.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D float32 array with small positive values close to zero
    x = np.array([[1e-4, 1e-3], [1e-2, 5e-2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D float64 array with int64 values
    x = np.array([4, 6, 7, 10], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.gammaln"] = gammaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gammaln'.")


check_valid('jax.scipy.special.gammaln', generated_inputs['jax.scipy.special.gammaln'], lib="jax", suffix=0)

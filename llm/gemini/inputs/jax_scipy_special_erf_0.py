
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def erf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with mixed positive and negative values
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: Scalar (0D tensor) float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float64 array
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float32 array
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: float16 array
    x = np.array([[-3.0, -0.5], [0.5, 3.0]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Integer array (JAX promotes real-valued ints to floats)
    x = np.array([-5, -2, 0, 1, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with extreme values (testing asymptotes)
    x = np.array([-1e6, -10.0, 0.0, 10.0, 1e6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array of all zeros
    x = np.zeros((4, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D float32 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.erf"] = erf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.erf'.")


check_valid('jax.scipy.special.erf', generated_inputs['jax.scipy.special.erf'], lib="jax", suffix=0)

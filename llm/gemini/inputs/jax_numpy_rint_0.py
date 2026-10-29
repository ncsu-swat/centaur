
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rint_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative half-way values
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar float32 array
    x = np.array(1.7, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float64 array
    x = np.array([[-10.2, 3.8], [0.1, -4.9]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D int32 array
    x = np.arange(-12, 12, dtype=np.int32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D complex64 array
    x = np.array([-2.5 + 3.5j, 4.5 - 0.5j, 1.2 + 2.8j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D complex128 array
    x = np.array(
        [[1.1 + 2.2j, -3.3 - 4.4j], [5.5 + 6.5j, -7.5 + 8.5j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D float32 array
    x = np.random.randn(2, 3, 4, 5).astype(np.float32) * 100.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D float16 array
    x = np.array([-1.2, 0.0, 1.8, 2.3], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D float32 array with special float values
    x = np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D int64 array
    x = np.array([[-100, 200, -300], [400, -500, 600]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.rint"] = rint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rint'.")


check_valid('jax.numpy.rint', generated_inputs['jax.numpy.rint'], lib="jax", suffix=0)

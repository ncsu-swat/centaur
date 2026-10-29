
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def divmod_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive float
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = 3.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D float32 array with negative and positive values
    x1 = np.array([-5.5, -2.1, 0.0, 3.7, 8.9], dtype=np.float32)
    x2 = 2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D int64 array, negative float
    x1 = np.array([[-15, 23], [42, -50]], dtype=np.int64)
    x2 = -4.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D float64 array
    x1 = np.array([[1.5, 2.7, 3.9], [4.1, 5.5, 6.2]], dtype=np.float64)
    x2 = 1.25
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D int32 array
    x1 = np.arange(1, 25, dtype=np.int32).reshape(2, 3, 4)
    x2 = 5.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar array
    x1 = np.array(42.0, dtype=np.float32)
    x2 = 5.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float32 array
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D int16 array with large values
    x1 = np.array([-1000, -500, 0, 500, 1000], dtype=np.int16)
    x2 = 120.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D uint8 array
    x1 = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
    x2 = 16.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D float64 array with fractions
    x1 = np.array([0.1, 0.25, 0.75, 1.33, 2.67], dtype=np.float64)
    x2 = -0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D float32 array
    x1 = np.ones((3, 2, 2), dtype=np.float32) * 7.5
    x2 = 2.2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.divmod_3"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_3'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_3'], lib="jax", suffix=3)

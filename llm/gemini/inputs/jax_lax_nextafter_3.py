
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nextafter_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array moving towards a positive float
    x1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    x2 = 2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D float32 array with negative values moving towards zero
    x1 = np.array([-3.0, -2.0, -1.0, -0.5], dtype=np.float32)
    x2 = 0.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float32 array moving downwards
    x1 = np.ones((2, 3), dtype=np.float32)
    x2 = -1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D float32 scalar array moving upwards
    x1 = np.array(0.0, dtype=np.float32)
    x2 = 1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float64 array moving towards a large float
    x1 = np.zeros((2, 2, 2), dtype=np.float64)
    x2 = 100.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 1D float32 array with infinities and extreme values
    x1 = np.array([float("inf"), float("-inf"), 1e30, -1e30], dtype=np.float32)
    x2 = 0.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float32 array moving towards a fractional float
    x1 = np.random.randn(1, 2, 3, 4).astype(np.float32)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D float32 array near subnormal/zero boundary
    x1 = np.array([0.0, -0.0, 1e-45, -1e-45], dtype=np.float32)
    x2 = 1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D float64 array moving towards negative infinity
    x1 = np.array([[10.0, -20.0], [30.0, -40.0]], dtype=np.float64)
    x2 = float("-inf")
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D float64 array moving towards positive infinity
    x1 = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    x2 = float("inf")
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D float32 array moving towards -10.0
    x1 = np.linspace(-5.0, 5.0, 12, dtype=np.float32).reshape(2, 2, 3)
    x2 = -10.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.lax.nextafter_3"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.nextafter_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.nextafter_3'.")


check_valid('jax.lax.nextafter', generated_inputs['jax.lax.nextafter_3'], lib="jax", suffix=3)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ptp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=0, keepdims=False
    a = np.array([-5.0, 0.0, 15.5, -2.3, 10.0], dtype=np.float32)
    axis = 0
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 2: 2D array, int32 with negative values, axis=1, keepdims=True
    a = np.array([[1, -3, 5, 2], [-4, 6, 8, 1], [7, 9, -3, 4]], dtype=np.int32)
    axis = 1
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 3: 2D array, float64, axis=0, keepdims=False
    a = np.random.randn(5, 4).astype(np.float64)
    axis = 0
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 4: 3D array, float32, axis=2, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 5: 3D array, int64, negative axis, keepdims=False
    a = np.random.randint(-100, 100, size=(4, 5, 6)).astype(np.int64)
    axis = -1
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 6: 4D array, float32, axis=1, keepdims=True
    a = np.random.randn(2, 4, 3, 5).astype(np.float32)
    axis = 1
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 7: 1D array, int16, negative axis, keepdims=True
    a = np.array([-100, 200, -300, 400], dtype=np.int16)
    axis = -1
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 8: 2D array, uint8, axis=0, keepdims=True
    a = np.array([[10, 50, 100], [200, 150, 80]], dtype=np.uint8)
    axis = 0
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 9: 3D array, float64, axis=-2, keepdims=False
    a = np.random.randn(3, 4, 5).astype(np.float64)
    axis = -2
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 10: 4D array, int32, axis=3, keepdims=False
    a = np.random.randint(0, 50, size=(2, 2, 3, 4)).astype(np.int32)
    axis = 3
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    return list_of_inputs


generated_inputs["jax.numpy.ptp_1"] = ptp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ptp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ptp_1'.")


check_valid('jax.numpy.ptp', generated_inputs['jax.numpy.ptp_1'], lib="jax", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cummin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default axis 0, reverse=False
    operand = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.0], dtype=np.float32)
    axis = 0
    reverse = False
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 2: 1D int32 array with negative values, reverse=True
    operand = np.array([-5, 10, -2, 0, 7, -8, 3], dtype=np.int32)
    axis = 0
    reverse = True
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 3: 2D float64 array, axis 0, reverse=False
    operand = np.array(
        [[1.5, -2.0, 3.5], [4.0, 0.5, -1.0]], dtype=np.float64
    )
    axis = 0
    reverse = False
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 4: 2D int64 array, axis 1, reverse=True
    operand = np.array([[10, 20, 5, 15], [8, 3, 12, 1]], dtype=np.int64)
    axis = 1
    reverse = True
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 5: 3D float32 array, axis 2, reverse=False
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    reverse = False
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 6: 3D float32 array, axis 1, reverse=True
    operand = np.random.randn(3, 4, 2).astype(np.float32)
    axis = 1
    reverse = True
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 7: 4D float32 array, axis 0, reverse=False
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = 0
    reverse = False
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 8: 1D int32 array, axis 0, reverse=False
    operand = np.array([100, 50, 75, 25, 60], dtype=np.int32)
    axis = 0
    reverse = False
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 9: 2D float64 array, axis 0, reverse=True
    operand = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    axis = 0
    reverse = True
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 10: 4D int64 array, axis 3, reverse=True
    operand = np.random.randint(-100, 100, size=(2, 3, 2, 4)).astype(np.int64)
    axis = 3
    reverse = True
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 11: 1D float32 array with inf and -inf
    operand = np.array([np.inf, 2.0, -1.0, -np.inf, 5.0], dtype=np.float32)
    axis = 0
    reverse = False
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    # Input 12: 2D int32 array, axis 1, reverse=False
    operand = np.array([[-10, 20, -30], [5, -5, 0]], dtype=np.int32)
    axis = 1
    reverse = False
    list_of_inputs.append(
        {"operand": operand, "axis": axis, "reverse": reverse}
    )

    return list_of_inputs


generated_inputs["jax.lax.cummin"] = cummin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cummin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cummin'.")


check_valid('jax.lax.cummin', generated_inputs['jax.lax.cummin'], lib="jax", suffix=0)

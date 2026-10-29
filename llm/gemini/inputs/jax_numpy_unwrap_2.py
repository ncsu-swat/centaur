
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unwrap_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with standard 2*pi period
    p = np.array([0.1, 0.5, 3.0, -3.1, -2.8], dtype=np.float32)
    discont = np.array(np.pi, dtype=np.float32)
    axis = -1
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 2: 1D float64 array in degrees with period 360
    p = np.array(
        [76.0, 133.0, 179.0, -157.0, -130.0, -127.0, -121.0], dtype=np.float64
    )
    discont = np.array(180.0, dtype=np.float64)
    axis = 0
    period = np.array(360.0, dtype=np.float64)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 3: 2D float32 array unwrapped along axis 0
    p = np.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [-3.0, -2.0, -1.0]],
        dtype=np.float32,
    )
    discont = np.array(np.pi, dtype=np.float32)
    axis = 0
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 4: 2D float32 array unwrapped along axis 1 with custom period 100.0
    p = np.array(
        [[10.0, 40.0, 80.0, 10.0], [90.0, 15.0, 50.0, 85.0]], dtype=np.float32
    )
    discont = np.array(50.0, dtype=np.float32)
    axis = 1
    period = np.array(100.0, dtype=np.float32)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 5: 3D float32 array unwrapped along axis -1
    p = np.linspace(0, 4 * np.pi, 24, dtype=np.float32).reshape(2, 3, 4)
    p = (p + np.pi) % (2 * np.pi) - np.pi
    discont = np.array(np.pi, dtype=np.float32)
    axis = -1
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 6: 3D float64 array with custom discont threshold along axis 1
    p = np.ones((3, 5, 2), dtype=np.float64) * np.pi
    discont = np.array(1.5, dtype=np.float64)
    axis = 1
    period = np.array(2 * np.pi, dtype=np.float64)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 7: 1D float32 array with strictly negative values
    p = np.array([-6.0, -5.5, -0.2, -6.1, -5.9], dtype=np.float32)
    discont = np.array(np.pi, dtype=np.float32)
    axis = -1
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 8: 2D float64 array unwrapped along axis -2 with period pi
    p = np.array([[0.1, 0.2], [3.0, 2.9], [0.3, 0.4]], dtype=np.float64)
    discont = np.array(np.pi / 2, dtype=np.float64)
    axis = -2
    period = np.array(np.pi, dtype=np.float64)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 9: 4D float32 array unwrapped along axis 2
    p = np.zeros((2, 2, 6, 2), dtype=np.float32)
    p[:, :, :, :] = np.linspace(-np.pi, np.pi, 6, dtype=np.float32).reshape(
        1, 1, 6, 1
    )
    discont = np.array(np.pi, dtype=np.float32)
    axis = 2
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    # Input 10: 1D float32 array with normalized period 1.0
    p = np.array([0.1, 0.4, 0.9, 0.2, 0.7, 0.1], dtype=np.float32)
    discont = np.array(0.5, dtype=np.float32)
    axis = 0
    period = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        {"p": p, "discont": discont, "axis": axis, "period": period}
    )

    return list_of_inputs


generated_inputs["jax.numpy.unwrap_2"] = unwrap_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unwrap_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unwrap_2'.")


check_valid('jax.numpy.unwrap', generated_inputs['jax.numpy.unwrap_2'], lib="jax", suffix=2)

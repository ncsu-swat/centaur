
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def expon_pdf_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like parameters
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 2: 1D array with negative values (below loc)
    x = np.array([-2.0, -1.0, 0.0, 0.5, 1.5], dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 3: 2D array, non-zero loc and fractional scale
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loc = 1.0
    scale = 0.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 4: 2D array with float64
    x = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float64)
    loc = -1.0
    scale = 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 5: 0D scalar tensor
    x = np.array(2.5, dtype=np.float32)
    loc = 0.0
    scale = 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 6: 3D array, large scale
    x = np.linspace(0, 10, 12, dtype=np.float32).reshape(2, 2, 3)
    loc = 2.0
    scale = 10.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 7: 4D array, small scale
    x = np.ones((2, 1, 2, 3), dtype=np.float32) * 0.5
    loc = 0.2
    scale = 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 8: Negative loc and scale > 1
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float64)
    loc = -3.0
    scale = 4.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 9: Large values in x
    x = np.array([100.0, 200.0, 500.0], dtype=np.float32)
    loc = 50.0
    scale = 50.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    loc = 0.0
    scale = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    # Input 11: 1D array with values exactly at loc
    x = np.array([3.14, 3.14, 4.14], dtype=np.float64)
    loc = 3.14
    scale = 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": float(loc), "scale": float(scale)})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.expon.pdf_2"] = expon_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.pdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.pdf_2'.")


check_valid('jax.scipy.stats.expon.pdf', generated_inputs['jax.scipy.stats.expon.pdf_2'], lib="jax", suffix=2)

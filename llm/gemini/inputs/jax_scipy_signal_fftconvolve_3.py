
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftconvolve_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, mode='full', convolve along axis 0
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([0.5, -1.0, 0.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "full", "axes": [0]}
        )
    )

    # Input 2: 1D float64 with negative values, mode='same'
    in1 = np.array([-2.5, 3.1, -0.4, 8.2, -1.1, 0.0], dtype=np.float64)
    in2 = np.array([1.0, -1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "same", "axes": [0]}
        )
    )

    # Input 3: 1D int32, mode='valid'
    in1 = np.array([10, -5, 3, 8, 2, -1, 4], dtype=np.int32)
    in2 = np.array([1, 2, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "valid", "axes": [0]}
        )
    )

    # Input 4: 2D float32, mode='full', both axes
    in1 = np.random.randn(4, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "full", "axes": [0, 1]}
        )
    )

    # Input 5: 2D float32, mode='same', 1 axis only
    in1 = np.random.randn(6, 6).astype(np.float32)
    in2 = np.random.randn(6, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "same", "axes": [1]}
        )
    )

    # Input 6: 2D float64, mode='valid', both axes
    in1 = np.random.randn(8, 7).astype(np.float64)
    in2 = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "valid", "axes": [0, 1]}
        )
    )

    # Input 7: 3D float32, mode='full', partial axes [0, 2]
    in1 = np.random.randn(3, 4, 5).astype(np.float32)
    in2 = np.random.randn(2, 4, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "full", "axes": [0, 2]}
        )
    )

    # Input 8: 3D complex64, mode='same', all axes
    in1 = (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
        np.complex64
    )
    in2 = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "same", "axes": [0, 1, 2]}
        )
    )

    # Input 9: 3D float32, mode='valid', single axis [1]
    in1 = np.random.randn(3, 8, 3).astype(np.float32)
    in2 = np.random.randn(3, 3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "valid", "axes": [1]}
        )
    )

    # Input 10: 4D float32, mode='full', axes [2, 3]
    in1 = np.random.randn(2, 2, 5, 5).astype(np.float32)
    in2 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "full", "axes": [2, 3]}
        )
    )

    # Input 11: 2D float64 with negative axis indexing, mode='same'
    in1 = np.random.randn(5, 5).astype(np.float64)
    in2 = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"in1": in1, "in2": in2, "mode": "same", "axes": [-2, -1]}
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.signal.fftconvolve_3"] = fftconvolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.fftconvolve_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.fftconvolve_3'.")


check_valid('jax.scipy.signal.fftconvolve', generated_inputs['jax.scipy.signal.fftconvolve_3'], lib="jax", suffix=3)

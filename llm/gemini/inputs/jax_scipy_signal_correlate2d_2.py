
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.lax
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=np.int32)


def correlate2d_inputs():
    list_of_inputs = []

    # Input 1: mode='full', float32, small square matrices
    in1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    in2 = np.array([[0.5, -1.0], [2.0, 1.5]], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: mode='same', float32, square arrays
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: mode='valid', float32, asymmetric dimensions
    in1 = np.random.randn(6, 4).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: mode='full', float64, rectangular inputs
    in1 = np.random.randn(4, 7).astype(np.float64)
    in2 = np.random.randn(3, 2).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: mode='same', float64, mixed precision values
    in1 = np.random.randn(7, 3).astype(np.float64)
    in2 = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: mode='valid', float32, same size inputs resulting in 1x1 output
    in1 = np.random.randn(4, 4).astype(np.float32)
    in2 = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: mode='full', 1-row arrays (horizontal strip)
    in1 = np.random.randn(1, 6).astype(np.float32)
    in2 = np.random.randn(1, 3).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: mode='same', 1-column filter (vertical strip)
    in1 = np.random.randn(8, 5).astype(np.float32)
    in2 = np.random.randn(3, 1).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: mode='valid', float64, larger matrices with negative values
    in1 = np.random.uniform(-5.0, 5.0, size=(10, 8)).astype(np.float64)
    in2 = np.random.uniform(-2.0, 2.0, size=(4, 3)).astype(np.float64)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: mode='full', 1x1 filter
    in1 = np.random.randn(3, 3).astype(np.float32)
    in2 = np.array([[-2.5]], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "boundary": "fill",
        "fillvalue": 0.0,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.correlate2d_2"] = correlate2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.correlate2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.correlate2d_2'.")


check_valid('jax.scipy.signal.correlate2d', generated_inputs['jax.scipy.signal.correlate2d_2'], lib="jax", suffix=2)

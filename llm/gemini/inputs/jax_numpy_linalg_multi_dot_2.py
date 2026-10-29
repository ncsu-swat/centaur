
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array(list(self), dtype=object)


def multi_dot_inputs():
    list_of_inputs = []

    # Input 1: 3 square matrices (float32)
    arrays = np.random.randn(3, 8, 8).astype(np.float32)
    precision = PrecisionTuple(("default", "default"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 2: 2 square matrices (float64)
    arrays = np.random.randn(2, 16, 16).astype(np.float64)
    precision = PrecisionTuple(("high", "high"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 3: 4 square matrices with negative values (float32)
    arrays = np.random.uniform(-5.0, 5.0, size=(4, 6, 6)).astype(np.float32)
    precision = PrecisionTuple(("highest", "highest"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 4: 5 small square matrices (float32)
    arrays = np.random.randn(5, 4, 4).astype(np.float32)
    precision = PrecisionTuple(("default", "high"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 5: 3 square matrices (float64)
    arrays = np.random.randn(3, 10, 10).astype(np.float64)
    precision = PrecisionTuple(("high", "default"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 6: 6 small square matrices (float32)
    arrays = np.random.uniform(-2.0, 2.0, size=(6, 3, 3)).astype(np.float32)
    precision = PrecisionTuple(("highest", "default"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 7: 2 large square matrices (float32)
    arrays = np.random.randn(2, 32, 32).astype(np.float32)
    precision = PrecisionTuple(("default", "highest"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 8: 4 square matrices (float64)
    arrays = np.random.randn(4, 5, 5).astype(np.float64)
    precision = PrecisionTuple(("high", "highest"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 9: 3 square matrices with mixed positive/negative values (float32)
    arrays = np.random.uniform(-10.0, 10.0, size=(3, 12, 12)).astype(
        np.float32
    )
    precision = PrecisionTuple(("highest", "high"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 10: 5 2x2 square matrices (float32)
    arrays = np.random.randn(5, 2, 2).astype(np.float32)
    precision = PrecisionTuple(("default", "default"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    # Input 11: 2 square matrices (float64)
    arrays = np.random.uniform(-1.0, 1.0, size=(2, 20, 20)).astype(np.float64)
    precision = PrecisionTuple(("high", "default"))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "precision": precision})
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.multi_dot_2"] = multi_dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.multi_dot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.multi_dot_2'.")


check_valid('jax.numpy.linalg.multi_dot', generated_inputs['jax.numpy.linalg.multi_dot_2'], lib="jax", suffix=2)

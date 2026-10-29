
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class RoundingMethodStr(str):

    def __new__(cls, name, val):
        obj = super().__new__(cls, name)
        obj._val = val
        return obj

    def __hash__(self):
        return hash(self._val)

    def __eq__(self, other):
        return self._val == other or super().__eq__(other)

    def __deepcopy__(self, memo):
        return self


AWAY_FROM_ZERO = RoundingMethodStr(
    "AWAY_FROM_ZERO", jax.lax.RoundingMethod.AWAY_FROM_ZERO.value
)
TO_NEAREST_EVEN = RoundingMethodStr(
    "TO_NEAREST_EVEN", jax.lax.RoundingMethod.TO_NEAREST_EVEN.value
)


def round_inputs():
    list_of_inputs = []

    # Input 1: 1D array with halfway values, float32, AWAY_FROM_ZERO
    x = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"x": x, "rounding_method": AWAY_FROM_ZERO}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with halfway values, float32, TO_NEAREST_EVEN
    x = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"x": x, "rounding_method": TO_NEAREST_EVEN}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array (3x3), float64, AWAY_FROM_ZERO
    x = np.array(
        [
            [-3.7, -2.5, -1.2],
            [0.1, 0.5, 0.9],
            [1.5, 2.5, 3.5],
        ],
        dtype=np.float64,
    )
    input_dict = {"x": x, "rounding_method": AWAY_FROM_ZERO}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array, float32, TO_NEAREST_EVEN
    x = np.array(4.5, dtype=np.float32)
    input_dict = {"x": x, "rounding_method": TO_NEAREST_EVEN}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array (2x3x4), float32, AWAY_FROM_ZERO
    x = np.linspace(-5.0, 5.0, 24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {"x": x, "rounding_method": AWAY_FROM_ZERO}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with float16 dtype, TO_NEAREST_EVEN
    x = np.array([-4.5, -3.5, -0.5, 0.5, 3.5, 4.5], dtype=np.float16)
    input_dict = {"x": x, "rounding_method": TO_NEAREST_EVEN}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array (2x2x2x2), float64, AWAY_FROM_ZERO
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "rounding_method": AWAY_FROM_ZERO}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with special values (nan, inf, signed zero), float32, TO_NEAREST_EVEN
    x = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "rounding_method": TO_NEAREST_EVEN}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array of shape (0,), float32, AWAY_FROM_ZERO
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x, "rounding_method": AWAY_FROM_ZERO}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with large floating point values, float64, TO_NEAREST_EVEN
    x = np.array(
        [1e8 + 0.5, -1e8 - 0.5, 1e12 + 0.5, -1e12 - 0.5], dtype=np.float64
    )
    input_dict = {"x": x, "rounding_method": TO_NEAREST_EVEN}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array (1x5) with negative fractional numbers, float32, AWAY_FROM_ZERO
    x = np.array([[-0.1, -0.5, -0.9, -1.5, -2.5]], dtype=np.float32)
    input_dict = {"x": x, "rounding_method": AWAY_FROM_ZERO}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 5D array (1, 2, 1, 3, 2), float32, TO_NEAREST_EVEN
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    input_dict = {"x": x, "rounding_method": TO_NEAREST_EVEN}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.round"] = round_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.round'.")


check_valid('jax.lax.round', generated_inputs['jax.lax.round'], lib="jax", suffix=0)

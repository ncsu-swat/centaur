
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D identical integer arrays
    a1 = np.array([1, 2, 3, -4], dtype=np.int32)
    a2 = np.array([1, 2, 3, -4], dtype=np.int32)
    equal_nan = False
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 2: 1D different shape integer arrays
    a1 = np.array([1, 2, 3], dtype=np.int64)
    a2 = np.array([1, 2], dtype=np.int64)
    equal_nan = False
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 3: 2D float arrays with NaNs, equal_nan=True
    a1 = np.array([[1.0, np.nan], [-3.5, 4.2]], dtype=np.float32)
    a2 = np.array([[1.0, np.nan], [-3.5, 4.2]], dtype=np.float32)
    equal_nan = True
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 4: 2D float arrays with NaNs, equal_nan=False
    a1 = np.array([[1.0, np.nan], [-3.5, 4.2]], dtype=np.float32)
    a2 = np.array([[1.0, np.nan], [-3.5, 4.2]], dtype=np.float32)
    equal_nan = False
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 5: 0D scalar arrays
    a1 = np.array(5.0, dtype=np.float64)
    a2 = np.array(5.0, dtype=np.float64)
    equal_nan = False
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 6: 3D boolean arrays
    a1 = np.array(
        [[[True, False], [False, True]], [[True, True], [False, False]]],
        dtype=np.bool_,
    )
    a2 = np.array(
        [[[True, False], [False, True]], [[True, True], [False, False]]],
        dtype=np.bool_,
    )
    equal_nan = True
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 7: 4D float arrays with differences
    a1 = np.zeros((2, 3, 4, 5), dtype=np.float32)
    a2 = np.zeros((2, 3, 4, 5), dtype=np.float32)
    a2[0, 1, 2, 3] = 1.0
    equal_nan = False
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 8: Empty arrays with same shape
    a1 = np.empty((0, 5), dtype=np.float64)
    a2 = np.empty((0, 5), dtype=np.float64)
    equal_nan = False
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 9: Empty arrays with different shapes
    a1 = np.empty((0, 5), dtype=np.int32)
    a2 = np.empty((0, 2), dtype=np.int32)
    equal_nan = True
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 10: 1D complex numbers with NaNs
    a1 = np.array([1.0 + 2.0j, -np.nan + 1.0j, 3.0 - 4.0j], dtype=np.complex64)
    a2 = np.array([1.0 + 2.0j, -np.nan + 1.0j, 3.0 - 4.0j], dtype=np.complex64)
    equal_nan = True
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 11: 2D arrays with infinities
    a1 = np.array([[np.inf, -np.inf], [0.0, 1.0]], dtype=np.float64)
    a2 = np.array([[np.inf, -np.inf], [0.0, 1.0]], dtype=np.float64)
    equal_nan = False
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    # Input 12: Arrays of different types but numerically equal values
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    equal_nan = True
    list_of_inputs.append(
        copy.deepcopy({"a1": a1, "a2": a2, "equal_nan": equal_nan})
    )

    return list_of_inputs


generated_inputs["jax.numpy.array_equal"] = array_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_equal'.")


check_valid('jax.numpy.array_equal', generated_inputs['jax.numpy.array_equal'], lib="jax", suffix=0)

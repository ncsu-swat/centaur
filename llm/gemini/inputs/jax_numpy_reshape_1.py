
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array reshaped to same length, order 'C', copy False
    a = np.arange(6, dtype=np.float32)
    shape = 6
    order = "C"
    copy_val = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 2: 2D float32 array flattened with -1, order 'F', copy True
    a = np.random.randn(2, 3).astype(np.float32)
    shape = -1
    order = "F"
    copy_val = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 3: 3D int32 array reshaped to 1D total size, order 'C', copy True
    a = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    shape = 8
    order = "C"
    copy_val = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 4: 4D float64 array reshaped to size 12, order 'F', copy False
    a = np.random.randn(1, 3, 2, 2).astype(np.float64)
    shape = 12
    order = "F"
    copy_val = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 5: 0D scalar array reshaped to 1D of size 1, order 'C', copy False
    a = np.array(42, dtype=np.int64)
    shape = 1
    order = "C"
    copy_val = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 6: 1D empty array of size 0, reshaped to 0, order 'C', copy True
    a = np.empty((0,), dtype=np.float32)
    shape = 0
    order = "C"
    copy_val = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 7: 2D empty array (0, 5) flattened with -1, order 'F', copy False
    a = np.empty((0, 5), dtype=np.int16)
    shape = -1
    order = "F"
    copy_val = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 8: 1D single-element array flattened with -1, order 'C', copy True
    a = np.array([3.14], dtype=np.float32)
    shape = -1
    order = "C"
    copy_val = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 9: 2D bool array reshaped to 4, order 'C', copy False
    a = np.array([[True, False], [False, True]], dtype=bool)
    shape = 4
    order = "C"
    copy_val = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 10: 2D complex64 array reshaped to size 6, order 'F', copy True
    a = (np.arange(6) + 1j * np.arange(6)).reshape(3, 2).astype(np.complex64)
    shape = 6
    order = "F"
    copy_val = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )

    # Input 11: 3D uint8 array flattened with -1, order 'F', copy False
    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    shape = -1
    order = "F"
    copy_val = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_val})
    )


    return list_of_inputs


generated_inputs["jax.numpy.reshape_1"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reshape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reshape_1'.")


check_valid('jax.numpy.reshape', generated_inputs['jax.numpy.reshape_1'], lib="jax", suffix=1)

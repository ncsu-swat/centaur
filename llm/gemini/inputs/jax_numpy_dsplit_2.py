
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dsplit_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 array split with single index
    ary = np.random.randn(2, 2, 4).astype(np.float32)
    indices_or_sections = [2]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 3D int32 array with negative values split at multiple indices
    ary = np.random.randint(-10, 10, size=(3, 3, 6), dtype=np.int32)
    indices_or_sections = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 4D float64 array split along axis 2
    ary = np.random.randn(2, 3, 6, 2).astype(np.float64)
    indices_or_sections = [2, 4]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 3D boolean array
    ary = np.random.choice([True, False], size=(1, 1, 2))
    indices_or_sections = [1]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 5D int64 array split at several indices
    ary = np.random.randint(0, 100, size=(2, 2, 8, 2, 2), dtype=np.int64)
    indices_or_sections = [2, 5, 7]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D complex64 array split at sorted indices
    real = np.random.randn(2, 2, 4).astype(np.float32)
    imag = np.random.randn(2, 2, 4).astype(np.float32)
    ary = (real + 1j * imag).astype(np.complex64)
    indices_or_sections = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 3D float32 array split with empty list
    ary = np.random.randn(2, 4, 3).astype(np.float32)
    indices_or_sections = []
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 4D uint8 array
    ary = np.random.randint(0, 256, size=(1, 2, 5, 3), dtype=np.uint8)
    indices_or_sections = [3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 3D float32 array split into three parts
    ary = np.random.randn(4, 4, 6).astype(np.float32)
    indices_or_sections = [2, 4]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 3D int16 array with larger depth axis
    ary = np.random.randint(-100, 100, size=(3, 2, 10), dtype=np.int16)
    indices_or_sections = [2, 4, 6, 8]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 3D float64 array split with single mid index
    ary = np.random.randn(2, 2, 2).astype(np.float64)
    indices_or_sections = [1]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.dsplit_2"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dsplit_2'.")


check_valid('jax.numpy.dsplit', generated_inputs['jax.numpy.dsplit_2'], lib="jax", suffix=2)

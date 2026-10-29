
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dsplit_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 array, single split point
    ary = np.random.randn(2, 2, 4).astype(np.float32)
    indices_or_sections = (2,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 3D int32 array, multiple split points
    ary = np.random.randint(-10, 10, size=(1, 3, 6)).astype(np.int32)
    indices_or_sections = (2, 4)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 4D float64 array, three split points
    ary = np.random.randn(3, 1, 8, 2).astype(np.float64)
    indices_or_sections = (1, 4, 6)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 3D float32 array, multiple split points
    ary = np.random.randn(2, 4, 5).astype(np.float32)
    indices_or_sections = (1, 2, 3, 4)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 3D int64 array, split points including 0
    ary = np.random.randint(-5, 5, size=(4, 4, 4)).astype(np.int64)
    indices_or_sections = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D float64 array, empty tuple for no split points
    ary = np.random.randn(1, 1, 3).astype(np.float64)
    indices_or_sections = ()
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 5D int64 array, two split points
    ary = np.random.randint(-100, 100, size=(2, 3, 10, 2, 2)).astype(np.int64)
    indices_or_sections = (3, 7)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 3D float32 array with negative values
    ary = np.linspace(-5.0, 5.0, num=36).reshape(3, 2, 6).astype(np.float32)
    indices_or_sections = (3,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 3D float32 array, evenly spaced split points
    ary = np.random.randn(2, 2, 8).astype(np.float32)
    indices_or_sections = (2, 4, 6)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 3D int32 array, split in half
    ary = np.random.randint(-50, 50, size=(5, 5, 2), dtype=np.int32)
    indices_or_sections = (1,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 4D float64 array, asymmetric split points
    ary = np.random.randn(1, 2, 6, 3).astype(np.float64)
    indices_or_sections = (2, 5)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.dsplit_3"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dsplit_3'.")


check_valid('jax.numpy.dsplit', generated_inputs['jax.numpy.dsplit_3'], lib="jax", suffix=3)

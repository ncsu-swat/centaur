
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def broadcast_shapes_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar shape
    list_of_inputs.append(copy.deepcopy({"shapes": ()}))

    # Input 2: 1D singleton shape
    list_of_inputs.append(copy.deepcopy({"shapes": (1,)}))

    # Input 3: 1D shape
    list_of_inputs.append(copy.deepcopy({"shapes": (5,)}))

    # Input 4: 2D singleton shape
    list_of_inputs.append(copy.deepcopy({"shapes": (1, 1)}))

    # Input 5: 2D shape
    list_of_inputs.append(copy.deepcopy({"shapes": (3, 4)}))

    # Input 6: 3D shape with singleton dimensions
    list_of_inputs.append(copy.deepcopy({"shapes": (1, 10, 1)}))

    # Input 7: 3D standard shape
    list_of_inputs.append(copy.deepcopy({"shapes": (2, 3, 4)}))

    # Input 8: 1D zero-size dimension
    list_of_inputs.append(copy.deepcopy({"shapes": (0,)}))

    # Input 9: 2D zero-size dimension
    list_of_inputs.append(copy.deepcopy({"shapes": (0, 5)}))

    # Input 10: 3D zero-size dimension
    list_of_inputs.append(copy.deepcopy({"shapes": (2, 0, 3)}))

    # Input 11: 4D shape
    list_of_inputs.append(copy.deepcopy({"shapes": (1, 2, 3, 4)}))

    # Input 12: 5D shape
    list_of_inputs.append(copy.deepcopy({"shapes": (2, 3, 4, 5, 6)}))

    return list_of_inputs


generated_inputs["jax.numpy.broadcast_shapes_1"] = broadcast_shapes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_shapes_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_shapes_1'.")


check_valid('jax.numpy.broadcast_shapes', generated_inputs['jax.numpy.broadcast_shapes_1'], lib="jax", suffix=1)

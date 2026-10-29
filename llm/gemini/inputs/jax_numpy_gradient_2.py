
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D array with uniform scalar spacing
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=np.float32)
    varargs = np.array(1.0, dtype=np.float32)
    axis = (0,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 2: 1D array with non-uniform coordinate array
    f = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float64)
    varargs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    axis = (0,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 3: 2D array, gradient along axis 0 with scalar spacing
    f = np.arange(12, dtype=np.float32).reshape((3, 4))
    varargs = np.array(0.5, dtype=np.float32)
    axis = (0,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 4: 2D array, gradient along axis 1 with scalar spacing
    f = np.array(
        [[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0], [0.0, 2.0, 4.0]], dtype=np.float32
    )
    varargs = np.array(2.0, dtype=np.float32)
    axis = (1,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 5: 2D array with 1D coordinate spacing along axis 1
    f = np.random.randn(4, 5).astype(np.float64)
    varargs = np.array([0.1, 0.4, 0.8, 1.3, 2.0], dtype=np.float64)
    axis = (1,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 6: 2D array with 1D coordinate spacing along axis 0
    f = np.random.randn(5, 3).astype(np.float32)
    varargs = np.array([1.0, 2.5, 4.0, 6.0, 8.5], dtype=np.float32)
    axis = (0,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 7: 2D array, gradient along multiple axes (0, 1) with scalar spacing
    f = np.sin(np.linspace(0, np.pi, 20, dtype=np.float32)).reshape((4, 5))
    varargs = np.array(0.25, dtype=np.float32)
    axis = (0, 1)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 8: 3D array, gradient along axis 2
    f = np.random.randn(3, 4, 5).astype(np.float32)
    varargs = np.array(1.5, dtype=np.float32)
    axis = (2,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 9: 3D array, gradient along axes (0, 2)
    f = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    varargs = np.array(1.0, dtype=np.float64)
    axis = (0, 2)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 10: 4D array, gradient along negative axis
    f = np.random.randn(2, 3, 4, 5).astype(np.float32)
    varargs = np.array(0.1, dtype=np.float32)
    axis = (-1,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    # Input 11: 1D array with negative values and irregular spacing
    f = np.array([-10.0, -5.0, 0.0, 5.0, 12.0], dtype=np.float32)
    varargs = np.array([-2.0, -1.0, 0.5, 1.0, 3.0], dtype=np.float32)
    axis = (0,)
    edge_order = None
    list_of_inputs.append(
        copy.deepcopy(
            {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.gradient_2"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_2'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_2'], lib="jax", suffix=2)

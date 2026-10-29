
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class TensorList(list):

    @property
    def shape(self):
        return (len(self), *self[0].shape)

    @property
    def dtype(self):
        return self[0].dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return len(self) * self[0].size

    def min(self):
        return min(x.min() for x in self)

    def max(self):
        return max(x.max() for x in self)


def concat_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays along axis 0
    arrays = TensorList(
        [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 0}))

    # Input 2: 1D int32 arrays with negative axis -1
    arrays = TensorList(
        [
            np.array([10, 20, 30], dtype=np.int32),
            np.array([40, 50, 60], dtype=np.int32),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": -1}))

    # Input 3: 2D float64 arrays along axis 0
    arrays = TensorList(
        [
            np.zeros((2, 4), dtype=np.float64),
            np.ones((2, 4), dtype=np.float64),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 0}))

    # Input 4: 2D float32 arrays along axis 1
    arrays = TensorList(
        [
            np.random.randn(3, 4).astype(np.float32),
            np.random.randn(3, 4).astype(np.float32),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 1}))

    # Input 5: 2D int64 arrays with negative axis -2
    arrays = TensorList(
        [
            np.ones((3, 5), dtype=np.int64),
            np.zeros((3, 5), dtype=np.int64),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": -2}))

    # Input 6: 3D boolean arrays along axis 2
    arrays = TensorList(
        [
            np.zeros((2, 3, 4), dtype=bool),
            np.ones((2, 3, 4), dtype=bool),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 2}))

    # Input 7: 3D float32 arrays (sequence of 3 arrays) along axis 0
    arrays = TensorList(
        [
            np.random.randn(2, 4, 3).astype(np.float32),
            np.random.randn(2, 4, 3).astype(np.float32),
            np.random.randn(2, 4, 3).astype(np.float32),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 0}))

    # Input 8: 4D int16 arrays along axis 3
    arrays = TensorList(
        [
            np.full((2, 3, 2, 4), 7, dtype=np.int16),
            np.full((2, 3, 2, 4), -3, dtype=np.int16),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 3}))

    # Input 9: 4D float32 arrays with negative axis -3
    arrays = TensorList(
        [
            np.random.randn(2, 3, 4, 5).astype(np.float32),
            np.random.randn(2, 3, 4, 5).astype(np.float32),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": -3}))

    # Input 10: 5D uint8 arrays along axis 1
    arrays = TensorList(
        [
            np.zeros((2, 3, 3, 2, 4), dtype=np.uint8),
            np.ones((2, 3, 3, 2, 4), dtype=np.uint8),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 1}))

    # Input 11: 2D complex64 arrays along axis 0
    arrays = TensorList(
        [
            np.array([[1.0 + 2.0j, 3.0 - 1.0j]], dtype=np.complex64),
            np.array([[0.0 + 0.0j, 5.0 + 5.0j]], dtype=np.complex64),
        ]
    )
    list_of_inputs.append(copy.deepcopy({"arrays": arrays, "axis": 0}))

    return list_of_inputs


generated_inputs["jax.numpy.concat_1"] = concat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.concat_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.concat_1'.")


check_valid('jax.numpy.concat', generated_inputs['jax.numpy.concat_1'], lib="jax", suffix=1)

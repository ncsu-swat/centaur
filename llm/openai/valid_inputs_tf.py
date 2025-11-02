generated_inputs = {}
import numpy as np
import tensorflow as tf

def tf_IndexedSlices_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 2D values and 1D indices
    values = np.array([[1., 2., 3.], [4., 5., 6.]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: 3D values with 1D indices
    values = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: 4D values with 1D indices
    values = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]])
    indices = np.array([0, 1])
    dense_shape = np.array([2, 2, 2, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: With negative values
    values = np.array([[-1., -2., -3.], [-4., -5., -6.]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: With float values
    values = np.array([[1.5, 2.5], [3.5, 4.5]])
    indices = np.array([0, 1])
    dense_shape = np.array([3, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Single element indices
    values = np.array([[1., 2., 3.]])
    indices = np.array([0])
    dense_shape = np.array([1, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: Large dense shape
    values = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]])
    indices = np.array([0, 1, 2])
    dense_shape = np.array([10, 5])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Different dtype values
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([0, 1])
    dense_shape = np.array([3, 2])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: Single dimension values with single index
    values = np.array([1., 2., 3., 4., 5.])
    indices = np.array([0])
    dense_shape = np.array([5])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: Mixed dimensions with different shapes
    values = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.], [13., 14., 15.]])
    indices = np.array([0, 1, 2, 3, 4])
    dense_shape = np.array([5, 3])
    
    input_dict = {
        "values": values,
        "indices": indices,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.IndexedSlices"] = tf_IndexedSlices_inputs()

import tensorflow as tf
import numpy as np

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []
    
    # Input 1: Valid - broadcast shapes with compatible dimensions
    shape_x = np.array([3, 4, 5], dtype=np.int32)
    shape_y = np.array([1, 4, 5], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: Valid - broadcast shapes with different dimensions
    shape_x = np.array([2, 3], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: Valid - broadcast shapes with one dimension being 1
    shape_x = np.array([1, 1, 1], dtype=np.int32)
    shape_y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: Valid - broadcast shapes with compatible dimensions (both are same)
    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: Valid - broadcast shapes with one dimension being 1 and another being 1
    shape_x = np.array([1, 1], dtype=np.int32)
    shape_y = np.array([1, 1], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Valid - broadcast shapes with different number of dimensions
    shape_x = np.array([2, 3], dtype=np.int32)
    shape_y = np.array([1, 3], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: Valid - broadcast shapes with negative values (valid)
    shape_x = np.array([-1, 2, 3], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Valid - broadcast shapes with zero dimension (valid)
    shape_x = np.array([0, 1], dtype=np.int32)
    shape_y = np.array([0, 2], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: Valid - broadcast shapes with same dimension but different values
    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: Valid - broadcast shapes with one dimension being 1 and others being 1
    shape_x = np.array([1, 1], dtype=np.int32)
    shape_y = np.array([1, 1], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape"] = tf_broadcast_dynamic_shape_inputs()

import numpy as np
import tensorflow as tf

def tf_compat_forward_compatible_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_dict = {
        "year": 2023,
        "month": 10,
        "day": 15
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    input_dict = {
        "year": 2024,
        "month": 1,
        "day": 1
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    input_dict = {
        "year": 2023,
        "month": 12,
        "day": 31
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    input_dict = {
        "year": 2022,
        "month": 6,
        "day": 15
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    input_dict = {
        "year": 2023,
        "month": 7,
        "day": 4
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    input_dict = {
        "year": 2023,
        "month": 2,
        "day": 29
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    input_dict = {
        "year": 2023,
        "month": 5,
        "day": 1
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    input_dict = {
        "year": 2024,
        "month": 3,
        "day": 10
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid
    input_dict = {
        "year": 2023,
        "month": 11,
        "day": 20
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid
    input_dict = {
        "year": 2023,
        "month": 9,
        "day": 5
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = tf_compat_forward_compatible_inputs()

import tensorflow as tf
import numpy as np

def tf_data_experimental_counter_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "start": np.int64(0),
        "step": np.int64(1),
        "dtype": np.int64
    }
    list_of_inputs.append(input_dict)
    
    # Input 2
    input_dict = {
        "start": np.int32(2),
        "step": np.int32(5),
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    # Input 3
    input_dict = {
        "start": np.int64(10),
        "step": np.int64(-1),
        "dtype": np.int64
    }
    list_of_inputs.append(input_dict)
    
    # Input 4
    input_dict = {
        "start": np.int32(-5),
        "step": np.int32(2),
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    # Input 5
    input_dict = {
        "start": np.int64(0),
        "step": np.int64(3),
        "dtype": np.int64
    }
    list_of_inputs.append(input_dict)
    
    # Input 6
    input_dict = {
        "start": np.int32(100),
        "step": np.int32(10),
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    # Input 7
    input_dict = {
        "start": np.int64(-10),
        "step": np.int64(-2),
        "dtype": np.int64
    }
    list_of_inputs.append(input_dict)
    
    # Input 8
    input_dict = {
        "start": np.int32(5),
        "step": np.int32(1),
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    # Input 9
    input_dict = {
        "start": np.int64(0),
        "step": np.int64(7),
        "dtype": np.int64
    }
    list_of_inputs.append(input_dict)
    
    # Input 10
    input_dict = {
        "start": np.int32(-3),
        "step": np.int32(1),
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.data.experimental.Counter"] = tf_data_experimental_counter_inputs()

import tensorflow as tf
import numpy as np
import copy

def tfexperimental_numpy_append_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6], [7, 8]])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6], [7, 8]])
    axis = 1
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = -1
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    values = np.array([[7, 8, 9], [10, 11, 12]])
    axis = 1
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    axis = -1
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6], [7, 8]])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.append"] = tfexperimental_numpy_append_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experiment_numpy_argmin_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with axis=0
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor with axis=1
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with axis=None
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with axis=0
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with axis=1
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with axis=2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D tensor with negative values
    a = np.array([-1, 2, -3, 4], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with negative values
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with axis=1 and negative values
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D tensor with axis=0 (corrected version)
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.argmin"] = tf_experiment_numpy_argmin_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_not_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    x = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    x = np.array([0, 1, 0], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    x = np.array([1, 0, -1], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    x = np.array([[-1, 0, 1], [0, -1, 1]], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.bitwise_not"] = tf_bitwise_not_inputs()

import tensorflow as tf
import numpy as np
import copy

def conj_inputs():
    list_of_inputs = []
    
    # Input 1: Real tensor
    x = np.array([1, 2, 3, 4])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Complex tensor with real and imaginary parts
    x = np.array([1+2j, 3+4j, 5+6j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Complex tensor with negative values
    x = np.array([-1-2j, -3-4j, -5-6j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Scalar complex number
    x = np.array(1+2j)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with complex numbers
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with complex numbers
    x = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Real tensor with floating point values
    x = np.array([1.5, 2.7, 3.9])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex tensor with zero imaginary part
    x = np.array([1+0j, 2+0j, 3+0j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex tensor with zero real part
    x = np.array([0+1j, 0+2j, 0+3j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mixed complex and real tensor
    x = np.array([1+0j, 2.5+3j, 4+5j, 6.7+8j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.conj"] = conj_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experiment_numpy_cumsum_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = 0
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([1, 2, 3, 4], dtype=np.float64)
    axis = 0
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = 1
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
    axis = 0
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis = -1
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    axis = 0
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis = 2
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    axis = 1
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([[[1, -2], [3, 4]], [[5, -6], [7, 8]]], dtype=np.float32)
    axis = -1
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([[-1, -2, -3], [4, -5, -6]], dtype=np.int64)
    axis = 1
    dtype = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.cumsum"] = tf_experiment_numpy_cumsum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    v = np.array([1, 2, 3, 4])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    v = np.array([1, 2, 3, 4])
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    v = np.array([[1, 2], [3, 4]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    v = np.array([1, 2, 3, 4])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    v = np.array([[1, 2, 3], [4, 5, 6]])
    k = 2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    v = np.array([[1, 2], [3, 4]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -2
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.diag"] = tf_experimental_numpy_diag_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experiment_numpy_fix_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.2, -2.7, 3.9])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([[1.5, -2.3], [3.7, -4.1]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([[[1.2, 2.8], [3.1, 4.9]], [[5.7, 6.3], [7.4, 8.6]]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([0.0, -1.5, 2.3])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([-0.9, 0.8, -1.2])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([-1.5, -2.5, -3.5])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([0.0, 0.0, 0.0])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[1.5, -2.3], [3.7, -4.1]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-1.9, 2.1], [-3.2, 4.3]])
    input_dict = {
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.fix"] = tf_experiment_numpy_fix_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_floor_divide_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    x2 = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x1 = np.array([1, 2, 3], dtype=np.float32)
    x2 = np.array([2, 4, -6], dtype=np.float32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x1 = np.array([[1.5, 2.7], [3.9, 4.1]], dtype=np.float64)
    x2 = np.array([[2.3, 4.5], [6.7, 8.9]], dtype=np.float64)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x1 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    x2 = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x1 = np.array([[-1.5, -2.7], [-3.9, -4.1]], dtype=np.float64)
    x2 = np.array([[2.3, 4.5], [6.7, 8.9]], dtype=np.float64)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = np.array([[2, 3], [4, 5]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([2, 4, -6], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x1 = np.array([[1.5, 2.7], [3.9, 4.1]], dtype=np.float64)
    x2 = np.array([[2.3, 4.5], [6.7, 8.9]], dtype=np.float64)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_floor_divide_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_isinf_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.0, 2.0, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-np.inf, 0.0, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([np.inf, np.nan, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([[np.inf, 1.0], [2.0, -np.inf]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[-np.inf, np.inf], [np.nan, 1.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[[np.inf, 1.0], [2.0, -np.inf]], [[3.0, np.nan], [4.0, 5.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([1.0, 2.0, np.inf, np.nan, -np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([0.0, 1.0, 2.0, np.inf, -np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[np.inf, np.nan], [1.0, 2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-np.inf, np.inf, np.nan], [1.0, 2.0, 3.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isinf"] = tf_isinf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experiment_numpy_isnan_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.0, np.nan, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([[1.0, np.nan], [3.0, np.nan]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([[[1.0, np.nan], [3.0, np.nan]], [[4.0, np.nan], [5.0, np.nan]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([np.nan, np.nan, np.nan])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - negative values
    x = np.array([-1.0, np.nan, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - mixed values
    x = np.array([1.0, np.nan, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - zero values
    x = np.array([0.0, np.nan, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - float infinity values
    x = np.array([np.inf, np.nan, -np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - integer values
    x = np.array([1, np.nan, 3])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isnan"] = tf_experiment_numpy_isnan_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_isposinf_inputs():
    list_of_inputs = []
    
    # Input 1: 1D array with positive infinity
    x = np.array([1.0, np.inf, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array with positive infinity
    x = np.array([[1.0, np.inf], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D array with negative infinity
    x = np.array([-np.inf, 1.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array with mixed infinities
    x = np.array([[np.inf, -np.inf], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar with positive infinity
    x = np.array(1.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: scalar with negative infinity
    x = np.array(-1.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D array with float values including infinities
    x = np.array([np.inf, np.nan, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D array with positive infinity
    x = np.array([[[np.inf, 1.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 1D array with all positive infinity values
    x = np.array([np.inf, np.inf, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D array with all negative infinity values
    x = np.array([[-np.inf, -np.inf], [-np.inf, -np.inf]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isposinf"] = generate_isposinf_inputs()

import numpy as np
import tensorflow as tf

def kron_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([7, 8])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[9, 10], [11, 12]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid - negative values
    a = np.array([[-1, 2], [3, -4]])
    b = np.array([[5, -6], [-7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid - floating point
    a = np.array([1.5, 2.7])
    b = np.array([3.1, 4.2])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid - mixed types
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid - scalar
    a = np.array(5)
    b = np.array(2)
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid - 4D array
    a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    b = np.array([[17, 18], [19, 20]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.kron"] = kron_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_moveaxis_inputs():
    list_of_inputs = []
    
    # Input 1
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = -2
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -2
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 1
    destination = -2
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -2
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = 2
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = -3
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.moveaxis"] = tf_moveaxis_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_nanprod_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with NaN values
    a = np.array([[1., 2., np.nan], [4., 5., 6.]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with NaN values
    a = np.array([[[1., 2.], [np.nan, 4.]], [[5., 6.], [7., np.nan]]])
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with NaN values
    a = np.array([1., 2., np.nan, 4., 5.])
    input_dict = {
        "a": a,
        "axis": None,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with all NaN values in one axis
    a = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with negative values
    a = np.array([-1., -2., 3., -4.])
    input_dict = {
        "a": a,
        "axis": None,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with mixed values including NaN
    a = np.array([[1., np.nan, 3.], [4., 5., np.nan]])
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": None,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with multiple NaN values
    a = np.array([[[np.nan, 2.], [3., 4.]], [[5., 6.], [np.nan, 8.]]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with large values
    a = np.array([100., 200., 300., np.nan])
    input_dict = {
        "a": a,
        "axis": None,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with zero values
    a = np.array([[1., 0.], [3., 4.]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with float dtype
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float64,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.nanprod"] = generate_nanprod_inputs()

import numpy as np
import tensorflow as tf

def tf_promote_types_inputs():
    list_of_inputs = []
    
    # Input 1
    type1 = np.int32
    type2 = np.float64
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 2
    type1 = np.uint8
    type2 = np.int16
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 3
    type1 = np.float32
    type2 = np.complex128
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 4
    type1 = np.bool_
    type2 = np.int32
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 5
    type1 = np.int64
    type2 = np.uint32
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 6
    type1 = np.float64
    type2 = np.int8
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 7
    type1 = np.uint16
    type2 = np.float32
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 8
    type1 = np.complex64
    type2 = np.int32
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 9
    type1 = np.bool_
    type2 = np.float64
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    # Input 10
    type1 = np.int16
    type2 = np.uint8
    input_dict = {
        "type1": type1,
        "type2": type2
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.promote_types"] = tf_promote_types_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_ravel_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([1, 2, 3, 4, 5])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([[1, 2], [3, 4], [5, 6]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([-1, -2, -3])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([[[[1, 2]], [[3, 4]]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([[[[1, 2, 3, 4]]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.ravel"] = tf_ravel_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signbit_inputs():
    list_of_inputs = []
    
    # Input 1, valid - tensor with negative values
    x = np.array([-1.0, 2.0, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - tensor with positive values
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - tensor with mixed values (positive and negative)
    x = np.array([-1.0, 2.0, -3.0, 4.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - scalar tensor
    x = np.array(5.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - 2D tensor with negative values
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - 3D tensor with negative values
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - tensor with zero values
    x = np.array([0.0, 1.0, -2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - tensor with very small values (close to zero)
    x = np.array([1e-10, -1e-10, 2e-10])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - tensor with large negative values
    x = np.array([-1000.0, 500.0, -2000.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - tensor with floating point values (positive and negative)
    x = np.array([1.5, -2.7, 3.9, -4.1])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.signbit"] = tf_signbit_inputs()

import tensorflow as tf
import numpy as np
import copy

def tril_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with k=0
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor with k=1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with k=-1
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with k=0
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with k=1
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with k=-1
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D tensor with k=0 (invalid - should be at least 2D)
    # m = np.array([1, 2, 3, 4])
    # k = 0
    # input_dict = {"m": m, "k": k}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D tensor with negative values and k=0
    m = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]])
    k = 0
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with k=1 and negative values
    m = np.array([[-1, 2], [4, -5]])
    k = 1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with k=-1 and negative values
    m = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]])
    k = -1
    input_dict = {"m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.tril"] = tril_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_vdot_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([-1, -2, -3])
    b = np.array([-4, -5, -6])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([0, 1, 2])
    b = np.array([3, 4, 5])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([-1, -2, -3, -4])
    b = np.array([-5, -6, -7, -8])
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.vdot"] = tf_vdot_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.5
    seed = 42
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    lower = 0.1
    upper = 0.9
    seed = 123
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    lower = 0.0
    upper = 0.5
    seed = 999
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    image = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)
    lower = 0.0
    upper = 1.0
    seed = 456
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.8
    seed = 789
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    lower = 0.0
    upper = 0.9
    seed = 111
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    image = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.3
    upper = 0.7
    seed = 222
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    lower = 0.0
    upper = 0.5
    seed = 333
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.9
    seed = 444
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    image = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    lower = 0.0
    upper = 0.3
    seed = 555
    
    input_dict = {
        "image": image,
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.random_contrast"] = generate_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_hue_inputs():
    list_of_inputs = []
    
    # Input 1
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.2
    seed = 42
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    image = np.array([[[0.5, 0.8, 0.3], [0.1, 0.9, 0.7]], [[0.2, 0.6, 0.4], [0.0, 0.5, 0.2]]], dtype=np.float32)
    max_delta = 0.1
    seed = 123
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    image = np.array([[[1.0, 0.0, 0.5], [0.2, 0.8, 0.9]], [[0.7, 0.3, 0.1], [0.6, 0.4, 0.0]]], dtype=np.float32)
    max_delta = 0.4
    seed = 50
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    image = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    max_delta = 0.3
    seed = 99
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=np.float32)
    max_delta = 0.0
    seed = 100
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    image = np.array([[[0.5, 0.8, 0.3], [0.1, 0.9, 0.7]], [[0.2, 0.6, 0.4], [0.0, 0.5, 0.2]]], dtype=np.float32)
    max_delta = 0.2
    seed = 200
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.5
    seed = 400
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=np.float32)
    max_delta = 0.4
    seed = 500
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    image = np.array([[[0.5, 0.8, 0.3], [0.1, 0.9, 0.7]], [[0.2, 0.6, 0.4], [0.0, 0.5, 0.2]]], dtype=np.float32)
    max_delta = 0.1
    seed = 600
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    image = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    max_delta = 0.4
    seed = 700
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor (height, width, channels)
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = np.array([2, 3], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor (batch, height, width, channels)
    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    seed = np.array([5, 7], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with different channel count
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    seed = np.array([1, 9], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with negative values
    image = np.array([[[1, -2], [-3, 4]], [[5, -6], [7, -8]]], dtype=np.int32)
    seed = np.array([10, 15], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with single channel
    image = np.array([[[[1]], [[2]]]], dtype=np.int32)
    seed = np.array([1, 1], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with multiple channels
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    seed = np.array([100, 200], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with mixed values
    image = np.array([[[[1, -2], [3, 4]], [[-5, 6], [-7, 8]]]], dtype=np.int32)
    seed = np.array([1000, 2000], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with zero values
    image = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.int32)
    seed = np.array([50, 50], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with large values
    image = np.array([[[[100, 200], [300, 400]], [[500, 600], [700, 800]]]], dtype=np.int32)
    seed = np.array([999, 888], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with large seed values
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    seed = np.array([999999, 888888], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down"] = tf_image_stateless_random_flip_up_down_inputs()

import tensorflow as tf
import copy
import numpy as np

def tf_image_transpose_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor (height, width, channels)
    image_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_3d),
        "name": "transpose_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor (batch, height, width, channels)
    image_4d = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d),
        "name": "transpose_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with negative values
    image_3d_neg = np.array([[[1, -2, 3], [-4, 5, -6]], [[7, -8, 9], [10, -11, 12]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_3d_neg),
        "name": "transpose_3d_neg"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with different channel count
    image_4d_diff_channels = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_4d_diff_channels),
        "name": "transpose_4d_diff_channels"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with floating point values
    image_3d_float = np.array([[[1.5, 2.7], [3.9, 4.1]], [[5.2, 6.8], [7.3, 8.6]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_3d_float),
        "name": "transpose_3d_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with single channel
    image_4d_single_channel = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d_single_channel),
        "name": "transpose_4d_single_channel"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with single channel
    image_3d_single_channel = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_3d_single_channel),
        "name": "transpose_3d_single_channel"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with different shapes
    image_4d_diff_shape = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d_diff_shape),
        "name": "transpose_4d_diff_shape"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with zero values
    image_3d_zero = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_3d_zero),
        "name": "transpose_3d_zero"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with large values
    image_4d_large_values = np.array([[[[100, 200], [300, 400]], [[500, 600], [700, 800]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d_large_values),
        "name": "transpose_4d_large_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_eigh_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor = np.array([[[3., 0., 0.], [0., 2., 0.], [0., 0., 1.]], 
                       [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor = np.array([[[1., 0.], [0., 1.]], [[2., 0.], [0., 2.]]], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    name = "test3"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor = np.array([[[1., 2.], [2., 1.]], [[3., 4.], [4., 3.]]], dtype=np.float32)
    name = "test4"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
    name = "test5"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor = np.array([[[1., -2.], [-2., 1.]], [[3., -4.], [-4., 3.]]], dtype=np.float32)
    name = "test6"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor = np.array([[1., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 3., 0.], [0., 0., 0., 4.]], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor = np.array([[[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]], [[4., 0., 0.], [0., 5., 0.], [0., 0., 6.]]], dtype=np.float32)
    name = "test8"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    name = "test9"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]], dtype=np.float32)
    name = "test10"
    
    input_dict = {
        "tensor": tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.eigh"] = tf_linalg_eigh_inputs()

import numpy as np
import tensorflow as tf

def tf_matrix_transpose_inputs():
    list_of_inputs = []
    
    # Input 1: Basic 2D tensor
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: Complex 2D tensor with conjugate
    a = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: 3D tensor with 2 batch dimensions
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: 4D tensor with 2 batch dimensions
    a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: 2D tensor with negative values
    a = np.array([[-1, -2, -3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: 3D tensor with negative values
    a = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: 3D tensor with complex values and conjugate
    a = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: 2D tensor with floating point values
    a = np.array([[1.5, 2.7], [3.9, 4.1]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: 2D tensor with zero values
    a = np.array([[0, 1], [2, 3]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: 3D tensor with zero values
    a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.linalg.matrix_transpose"] = tf_matrix_transpose_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_trace_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor (matrix)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor (3D matrix)
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor (4D matrix)
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with float values
    x = np.array([[1.5, 2.7], [3.1, 4.9]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "trace_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with negative values
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with mixed values
    x = np.array([[[1, -2, 3], [4, 5, -6], [7, 8, 9]], [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with zero values
    x = np.array([[0, 1], [2, 3]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with negative values
    x = np.array([[[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]], [[[-9, -10], [-11, -12]], [[-13, -14], [-15, -16]]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with float values
    x = np.array([[1.5, 2.7], [3.1, 4.9]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "trace_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with negative values
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.trace"] = tf_linalg_trace_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_atan2_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with positive values
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Mixed signs
    y = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    x = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: All negative values
    y = np.array([-1.0, -2.0], dtype=np.float32)
    x = np.array([-1.0, -1.0], dtype=np.float32)
    name = "test3"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different shapes (2D array)
    y = np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32)
    x = np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    name = "test4"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Zero values in x
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([1.0, 0.0], dtype=np.float32)
    name = "test5"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float64 type
    y = np.array([1.0, 2.0], dtype=np.float64)
    x = np.array([1.0, 1.0], dtype=np.float64)
    name = "test6"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large values
    y = np.array([100.0, 200.0], dtype=np.float32)
    x = np.array([50.0, 100.0], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Negative values in both arrays
    y = np.array([-1.0, -2.0], dtype=np.float32)
    x = np.array([-1.0, -2.0], dtype=np.float32)
    name = "test8"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Single element arrays
    y = np.array([1.0], dtype=np.float32)
    x = np.array([1.0], dtype=np.float32)
    name = "test9"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex numbers (as real arrays)
    y = np.array([1.0, 2.0], dtype=np.float32)
    x = np.array([1.0, 2.0], dtype=np.float32)
    name = "test10"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.atan2"] = tf_math_atan2_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_bessel_i1e_inputs():
    list_of_inputs = []
    
    # Input 1: Negative values
    x = np.array([-1., -0.5, 0.5, 1.], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Mixed values
    x = np.array([-2., -1., 0., 1., 2.], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Single element array
    x = np.array([5.], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Zero values
    x = np.array([0., 0., 0.], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Large values
    x = np.array([10., 20., 30.], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Small values
    x = np.array([0.001, 0.0001, 0.00001], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float values with negatives
    x = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Multi-dimensional array
    x = np.array([[-1., -0.5], [0.5, 1.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex values (as float)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Negative and positive mixed
    x = np.array([-2., 0., 2.], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.bessel_i1e"] = generate_bessel_i1e_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_cos_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with mixed values
    x = np.array([-1.5, 0.0, 1.5, 3.14], dtype=np.float64)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64 tensor
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128 tensor
    x = np.array([1+2j, 3+4j], dtype=np.complex128)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar tensor
    x = np.array(1.0, dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with negative values
    x = np.array([-5, -2.5, 0, 2.5, 5], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with negative values
    x = np.array([[-1, -2], [3, 4]], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with negative values
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with nan values
    x = np.array([np.nan, -np.inf, np.inf], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 tensor with zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.cos"] = generate_cos_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_cumprod_inputs():
    list_of_inputs = []
    
    # Input 1: Basic tensor with axis=0, exclusive=False, reverse=False
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = False
    name = "test1"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Basic tensor with axis=0, exclusive=True, reverse=False
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = True
    reverse = False
    name = "test2"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Basic tensor with axis=0, exclusive=False, reverse=True
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = True
    name = "test3"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Basic tensor with axis=0, exclusive=True, reverse=True
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = True
    reverse = True
    name = "test4"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with axis=1, exclusive=False, reverse=False
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = False
    reverse = False
    name = "test5"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with axis=1, exclusive=True, reverse=False
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = True
    reverse = False
    name = "test6"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with axis=1, exclusive=False, reverse=True
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = False
    reverse = True
    name = "test7"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with axis=1, exclusive=True, reverse=True
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = True
    reverse = True
    name = "test8"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with axis=0, exclusive=False, reverse=False
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = False
    name = "test9"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D tensor with axis=2, exclusive=True, reverse=True
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 2
    exclusive = True
    reverse = True
    name = "test10"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.cumprod"] = tf_math_cumprod_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_erf_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    x = np.array(0.5, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    x = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    x = np.array([[[1.0, 2.0], [0.0, -1.0]], [[2.0, 3.0], [-1.0, 0.0]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: mixed positive and negative values
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float64 tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: bfloat16 tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: tensor with zero values
    x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.erf"] = tf_math_erf_inputs()

import numpy as np
import tensorflow as tf

def floormod_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int64)
    name = "test2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, -6.0], dtype=np.float32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 4.0, -6.0], dtype=np.float64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    x = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int16)
    name = "test6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    y = np.array([[2.0, 4.0], [-6.0, 8.0]], dtype=np.float64)
    name = "test7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([2, 4, -6], dtype=np.int8)
    name = "test8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    y = np.array([[2, 3], [4, 5]], dtype=np.uint8)
    name = "test9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid
    x = np.array([[-1, -2], [-3, -4]], dtype=np.uint8)
    y = np.array([[2, 4], [-6, 8]], dtype=np.uint8)
    name = "test10"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.math.floormod"] = floormod_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_invert_permutation_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([0, 1, 2, 3], dtype=np.int64)
    name = "test2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([2, 0, 1], dtype=np.int64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([1, 3, 0, 2], dtype=np.int32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - single element permutation
    x = np.array([0], dtype=np.int32)
    name = "test6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - larger permutation
    x = np.array([5, 4, 3, 2, 1, 0], dtype=np.int64)
    name = "test7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - different shape (1D tensor with single element)
    x = np.array([0], dtype=np.int32)
    name = "test8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([2, 1, 0], dtype=np.int32)
    name = "test9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([4, 0, 2, 1, 3], dtype=np.int64)
    name = "test10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.invert_permutation"] = tf_math_invert_permutation_inputs()

import numpy as np
import tensorflow as tf

def tf_math_is_inf_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([5.0, np.inf, 6.8, np.inf], dtype=np.float64)
    name = "test1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2, valid
    x = np.array([-np.inf, 0.0, np.inf, 1.0], dtype=np.float32)
    name = "test2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3, valid
    x = np.array([np.inf], dtype=np.float64)
    name = "test3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4, valid
    x = np.array([-np.inf, np.inf, -np.inf], dtype=np.float32)
    name = "test4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "test5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6, valid
    x = np.array([np.nan, np.inf, np.nan], dtype=np.float32)
    name = "test6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7, valid
    x = np.array([[np.inf, 1.0], [2.0, np.inf]], dtype=np.float64)
    name = "test7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8, valid
    x = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    name = "test8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 9, valid
    x = np.array([-np.inf, -np.inf, np.inf], dtype=np.float64)
    name = "test9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    # Input 10, valid
    x = np.array([0.0, np.inf, 0.0], dtype=np.float32)
    name = "test10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.math.is_inf"] = tf_math_is_inf_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_lgamma_inputs():
    list_of_inputs = []
    
    # Input 1: Positive integers
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    input_dict = {"x": x, "name": "positive_integers"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Positive floats
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "name": "positive_floats"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Negative floats (with non-integer values)
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    input_dict = {"x": x, "name": "negative_floats"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Mixed positive and negative values
    x = np.array([0, 0.5, 1, -1, -2], dtype=np.float64)
    input_dict = {"x": x, "name": "mixed_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single element array
    x = np.array([3.5], dtype=np.float32)
    input_dict = {"x": x, "name": "single_element"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative integers
    x = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {"x": x, "name": "negative_integers"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large positive values
    x = np.array([10.5, 15.7, 20.2], dtype=np.float64)
    input_dict = {"x": x, "name": "large_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Small positive values
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"x": x, "name": "small_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Zero and negative values
    x = np.array([0, -0.5, -1.5], dtype=np.float64)
    input_dict = {"x": x, "name": "zero_and_negatives"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Float arrays with different shapes
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"x": x, "name": "multi_dimensional"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.lgamma"] = generate_lgamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_log1p_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    x = np.array(0.5, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with negative values
    x = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    x = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with complex numbers
    x = np.array([1+1j, 2+2j], dtype=np.complex64)
    input_dict = {
        "x": x,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with float64
    x = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar tensor with float32
    x = np.array(1.5, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with half precision
    x = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 1D tensor with bfloat16
    x = np.array([0.5, 1.0, 1.5], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D tensor with negative values
    x = np.array([-0.9, -0.5, -0.1], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.log1p"] = generate_log1p_inputs()

import tensorflow as tf
import copy

def tf_math_logical_not_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = tf.constant([True, False])
    name = "test1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = tf.constant([False, True, False])
    name = "test2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = tf.constant([[True, False], [False, True]])
    name = "test3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = tf.constant([[[True, False, True], [False, True, False]], [[True, True, False], [False, False, True]]])
    name = "test4"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = tf.constant([True])
    name = "test5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = tf.constant([False])
    name = "test6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = tf.constant([True, False, True, False, True])
    name = "test7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = tf.constant([[False, True], [True, False], [False, True]])
    name = "test8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = tf.constant([[[True, False], [False, True]]])
    name = "test9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = tf.constant([False, False, False, True])
    name = "test10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.logical_not"] = tf_math_logical_not_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_maximum_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([0., 0., 0., 0.], dtype=np.float32)
    y = np.array([-2., 0., 2., 5.], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-5., 0., 0., 0.], dtype=np.float32)
    y = np.array([-3.], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([1.5, 2.7, 3.1], dtype=np.float64)
    y = np.array([3.1, 2.7, 1.5], dtype=np.float64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([-1, -2, -3], dtype=np.int64)
    y = np.array([-3, -2, -1], dtype=np.int64)
    name = "test5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([0., 1., 2.], dtype=np.float32)
    y = np.array([2., 1., 0.], dtype=np.float32)
    name = "test6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([2.5, 3.7], dtype=np.float32)
    y = np.array([1.9, 4.1], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([0.5, 1.5], dtype=np.float64)
    y = np.array([1.5, 0.5], dtype=np.float64)
    name = "test8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int8)
    y = np.array([[-4, -3], [-2, -1]], dtype=np.int8)
    name = "test9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([3, 2, 1], dtype=np.int16)
    name = "test10"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.maximum"] = tf_math_maximum_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_polygamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([0, 1, 2], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "polygamma_1"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([0, 1], dtype=np.float64)
    x = np.array([1.0, 2.0], dtype=np.float64)
    name = "polygamma_2"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([3], dtype=np.float32)
    x = np.array([1.5], dtype=np.float32)
    name = "polygamma_3"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([5, 2], dtype=np.float64)
    x = np.array([1.0, 2.0], dtype=np.float64)
    name = "polygamma_4"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([10], dtype=np.float32)
    x = np.array([3.14], dtype=np.float32)
    name = "polygamma_5"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([0, 1, 2, 3], dtype=np.float64)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "polygamma_6"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([1], dtype=np.float32)
    x = np.array([0.5], dtype=np.float32)
    name = "polygamma_7"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([2], dtype=np.float64)
    x = np.array([1.0], dtype=np.float64)
    name = "polygamma_8"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([3], dtype=np.float32)
    x = np.array([2.5], dtype=np.float32)
    name = "polygamma_9"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([0], dtype=np.float64)
    x = np.array([1.0], dtype=np.float64)
    name = "polygamma_10"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.polygamma"] = tf_math_polygamma_inputs()

import numpy as np
import tensorflow as tf

def tf_math_real_inputs():
    list_of_inputs = []
    
    # Input 1: Real tensor with negative values
    input_tensor = np.array([-2.25, -3.25, -4.75], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_negative"}
    list_of_inputs.append(input_dict)
    
    # Input 2: Real tensor with positive values
    input_tensor = np.array([2.25, 3.25, 4.75], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_positive"}
    list_of_inputs.append(input_dict)
    
    # Input 3: Complex tensor with real part
    input_tensor = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_real"}
    list_of_inputs.append(input_dict)
    
    # Input 4: Complex tensor with imaginary part
    input_tensor = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_imaginary"}
    list_of_inputs.append(input_dict)
    
    # Input 5: Complex tensor with mixed real and imaginary parts
    input_tensor = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "complex_mixed"}
    list_of_inputs.append(input_dict)
    
    # Input 6: Real tensor with zero values
    input_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_zero"}
    list_of_inputs.append(input_dict)
    
    # Input 7: Real tensor with decimal values
    input_tensor = np.array([1.5, 2.7, 3.9], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_decimal"}
    list_of_inputs.append(input_dict)
    
    # Input 8: Real tensor with large values
    input_tensor = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_large"}
    list_of_inputs.append(input_dict)
    
    # Input 9: Real tensor with small values
    input_tensor = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_small"}
    list_of_inputs.append(input_dict)
    
    # Input 10: Real tensor with negative values
    input_tensor = np.array([-1.5, -2.7, -3.9], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "real_negative_decimal"}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.math.real"] = tf_math_real_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_bessel_j1_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    x = tf.constant(0.5, dtype=tf.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor
    x = tf.constant([0.5, 1., 2., 4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    x = tf.constant([[0.5, 1.], [2., 4.]], dtype=tf.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: negative values
    x = tf.constant([-0.5, -1., -2., -4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: mixed positive and negative values
    x = tf.constant([0.5, -1., 2., -4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor
    x = tf.constant([0.5, 1., 2., 4.], dtype=tf.float64)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar tensor with float64
    x = tf.constant(0.5, dtype=tf.float64)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: zero values
    x = tf.constant([0., 1., 2., 4.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: large values
    x = tf.constant([100., 200., 300., 400.], dtype=tf.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: small values
    x = tf.constant([0.001, 0.01, 0.1, 0.0001], dtype=tf.float32)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = generate_bessel_j1_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_squared_difference_inputs():
    list_of_inputs = []
    
    # Input 1: Basic tensor with positive values
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.float32)
    name = "input1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Tensor with negative values
    x = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float64)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.float64)
    name = "input2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Scalar tensors
    x = np.array(5.0, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    name = "input3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([2, 3, 4, 5], dtype=np.int32)
    name = "input4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex numbers
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    y = np.array([2+3j, 4+5j], dtype=np.complex64)
    name = "input5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Broadcasting tensors (different shapes)
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float32)
    name = "input6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Zero tensors
    x = np.array([[0, 1], [2, 3]], dtype=np.float32)
    y = np.array([[0, 1], [2, 3]], dtype=np.float32)
    name = "input7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Large difference (float64)
    x = np.array([[1000, 2000], [3000, 4000]], dtype=np.float64)
    y = np.array([[100, 200], [300, 400]], dtype=np.float64)
    name = "input8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large numbers (int64)
    x = np.array([[100, 200], [300, 400]], dtype=np.int64)
    y = np.array([[50, 100], [150, 200]], dtype=np.int64)
    name = "input9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mixed types (float32 and float64) - Removed to avoid error
    # x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # y = np.array([[2, 3], [4, 5]], dtype=np.float64)
    # name = "input10"
    
    # input_dict = {
    #     "x": x,
    #     "y": y,
    #     "name": name
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.squared_difference"] = tf_squared_difference_inputs()

import numpy as np
import tensorflow as tf

def tf_math_zero_fraction_inputs():
    list_of_inputs = []
    
    # Input 1: Empty tensor
    value = np.array([], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_1"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: All zeros tensor
    value = np.array([0, 0, 0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_2"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: Mixed values with some zeros
    value = np.array([1, 0, 3, 0, 5], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_3"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: Multi-dimensional tensor with zeros
    value = np.array([[1, 0], [0, 3]], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_4"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: All non-zero values
    value = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_5"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Negative values with zeros
    value = np.array([-1, 0, -3, 0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_6"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: Single element tensor
    value = np.array([0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_7"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Large tensor with many zeros
    value = np.random.choice([0, 1], size=(1000,), p=[0.5, 0.5])
    input_dict = {
        "value": value,
        "name": "test_8"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: Tensor with negative values
    value = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_9"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: Float tensor with NaN values
    value = np.array([np.nan, np.nan, 0], dtype=np.float32)
    input_dict = {
        "value": value,
        "name": "test_10"
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_crelu_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with positive and negative values
    features = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    axis = -1
    name = "test1"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with positive and negative values
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.float32)
    axis = -1
    name = "test2"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with positive and negative values
    features = np.array([1, -2, 3, -4, 5], dtype=np.float32)
    axis = -1
    name = "test3"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with positive and negative values
    features = np.array([[[[1, -2], [3, -4]], [[5, -6], [7, -8]]], [[[-9, 10], [-11, 12]], [[13, -14], [-15, 16]]]], dtype=np.float32)
    axis = -1
    name = "test4"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with all positive values
    features = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = -1
    name = "test5"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with all negative values
    features = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)
    axis = -1
    name = "test6"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with mixed values, different axis
    features = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    axis = 0
    name = "test7"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with mixed values, different axis
    features = np.array([1, -2, 3, -4, 5], dtype=np.float32)
    axis = 0
    name = "test8"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with positive and negative values, different axis
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.float32)
    axis = 0
    name = "test9"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with all positive values, different axis
    features = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    axis = 1
    name = "test10"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.crelu"] = generate_crelu_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_isotonic_regression_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_1 = np.array([[3, 1, 2], [1, 3, 4]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_1),
        "decreasing": True,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_2),
        "decreasing": True,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_3 = np.array([1, 2, 3, 4], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_3),
        "decreasing": False,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_4 = np.array([[5, 4, 3], [2, 1, 0]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_4),
        "decreasing": False,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_5 = np.array([[-1, -2, -3], [1, 2, 3]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_5),
        "decreasing": True,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_6),
        "decreasing": True,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_7 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_7),
        "decreasing": False,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_8 = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_8),
        "decreasing": True,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_9 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_9),
        "decreasing": True,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_10 = np.array([[5, 4, 3], [2, 1, 0]], dtype=np.float32)
    input_dict = {
        "inputs": tf.constant(input_10),
        "decreasing": False,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.isotonic_regression"] = tf_isotonic_regression_inputs()

import numpy as np
import tensorflow as tf

def tf_nn_log_poisson_loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    targets = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    log_input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    compute_full_loss = False
    name = "test1"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    targets = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    log_input = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    compute_full_loss = True
    name = "test2"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    targets = np.array([1, 2, 3, 4], dtype=np.float32)
    log_input = np.array([1, 2, 3, 4], dtype=np.float32)
    compute_full_loss = False
    name = "test3"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    targets = np.array([-1, -2, -3], dtype=np.float32)
    log_input = np.array([-1, -2, -3], dtype=np.float32)
    compute_full_loss = False
    name = "test4"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    targets = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    log_input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    compute_full_loss = True
    name = "test5"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    targets = np.array([[1, 2], [3, 4]], dtype=np.float32)
    log_input = np.array([[1, 2], [3, 4]], dtype=np.float32)
    compute_full_loss = True
    name = "test6"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    targets = np.array([1.5, 2.7, 3.1], dtype=np.float32)
    log_input = np.array([1.5, 2.7, 3.1], dtype=np.float32)
    compute_full_loss = False
    name = "test7"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    targets = np.array([0, 1, 2], dtype=np.float32)
    log_input = np.array([0, 1, 2], dtype=np.float32)
    compute_full_loss = True
    name = "test8"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid
    targets = np.array([2.3, 3.4, 4.5], dtype=np.float32)
    log_input = np.array([2.3, 3.4, 4.5], dtype=np.float32)
    compute_full_loss = True
    name = "test9"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid
    targets = np.array([5.6, 6.7, 7.8], dtype=np.float32)
    log_input = np.array([5.6, 6.7, 7.8], dtype=np.float32)
    compute_full_loss = False
    name = "test10"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.nn.log_poisson_loss"] = tf_nn_log_poisson_loss_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softmax_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    logits = np.array([-1, 0., 1.], dtype=np.float32)
    axis = -1
    name = "softmax_1"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    logits = np.array([[-1, 0., 1.], [-2, 1., 3.]], dtype=np.float32)
    axis = -1
    name = "softmax_2"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    logits = np.array([[-1, 0., 1.], [-2, 1., 3.], [-4, 5., 6.]], dtype=np.float32)
    axis = -1
    name = "softmax_3"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    logits = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = -1
    name = "softmax_4"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    logits = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    axis = -1
    name = "softmax_5"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    logits = np.array([[-1, 0., 1.], [-2, 1., 3.], [-4, 5., 6.]], dtype=np.float64)
    axis = -1
    name = "softmax_6"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    logits = np.array([-1, 0., 1., 2., 3., 4.], dtype=np.float32)
    axis = -1
    name = "softmax_7"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    logits = np.array([[-1, 0., 1.], [-2, 1., 3.], [-4, 5., 6.], [-7, 8., 9.]], dtype=np.float32)
    axis = -1
    name = "softmax_8"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    logits = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    axis = -1
    name = "softmax_9"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    logits = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], dtype=np.float32)
    axis = -1
    name = "softmax_10"
    input_dict = {
        "logits": logits,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.softmax"] = tf_nn_softmax_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_cosh_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with mixed values
    x = np.array([-9.5, -0.5, 0.0, 1.5, 2.5], dtype=np.float64)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: half tensor with complex values
    x = np.array([1+1j, 2+2j, 0+0j], dtype=np.complex64)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: bfloat16 tensor with large values
    x = np.array([100, 200, 300], dtype=np.float32)  # Note: bfloat16 not directly supported, but using float32 as proxy
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar tensor (1D)
    x = np.array([1.5], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with negative values
    x = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor (not just one dimension)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: complex128 tensor with real values
    x = np.array([1.0j, 2.0j], dtype=np.complex128)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with very small values
    x = np.array([0.001, 0.0001], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: tensor with zero values
    x = np.array([0, 0, 0], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = generate_cosh_inputs()

import numpy as np
import tensorflow as tf

def tf_raw_ops_div_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    input_dict = {
        "name": "div_1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 2: int32 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 4], [6, 8]], dtype=np.int32)
    input_dict = {
        "name": "div_2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 3: float64 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
    input_dict = {
        "name": "div_3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 4: complex64 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex64)
    input_dict = {
        "name": "div_4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 5: uint8 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[2, 4], [6, 8]], dtype=np.uint8)
    input_dict = {
        "name": "div_5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 6: int16 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[2, 4], [6, 8]], dtype=np.int16)
    input_dict = {
        "name": "div_6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 7: int64 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[2, 4], [6, 8]], dtype=np.int64)
    input_dict = {
        "name": "div_7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 8: float32 scalar tensor
    x = np.array(10.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    input_dict = {
        "name": "div_8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 9: float64 scalar tensor
    x = np.array(10.0, dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {
        "name": "div_9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 10: complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex128)
    input_dict = {
        "name": "div_10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.Div"] = tf_raw_ops_div_inputs()

import tensorflow as tf
import numpy as np
import copy

def elu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    feature = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "name": "el1",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    feature = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {
        "name": "el2",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    feature = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {
        "name": "el3",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    feature = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    input_dict = {
        "name": "el4",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    feature = np.array([[-1000.0], [-1.0]], dtype=np.float32)
    input_dict = {
        "name": "el5",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    feature = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "name": "el6",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    feature = np.array([1.0], dtype=np.float32)
    input_dict = {
        "name": "el7",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    feature = np.array([-1.0], dtype=np.float32)
    input_dict = {
        "name": "el8",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    feature = np.array([0.0], dtype=np.float32)
    input_dict = {
        "name": "el9",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    feature = np.array([[-1000.0, -100.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {
        "name": "el10",
        "features": feature
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = elu_inputs()

import numpy as np
import tensorflow as tf

def tf_raw_ops_fact_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 2, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 3, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 4, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 6, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 7, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 8, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 9, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    # Input 10, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_fact_inputs()

import numpy as np
import tensorflow as tf

def tf_greater_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([5.0, 4.0, 6.0], dtype=np.float32)
    y = np.array([5.0, 2.0, 5.0], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: int32 tensors
    x = np.array([5, 4, 6], dtype=np.int32)
    y = np.array([5, 2, 5], dtype=np.int32)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: int64 tensors
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([5, 10, 15], dtype=np.int64)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: float64 tensors
    x = np.array([1.5, 2.7, 3.1], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: uint8 tensors
    x = np.array([255, 128, 64], dtype=np.uint8)
    y = np.array([128, 64, 32], dtype=np.uint8)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: int8 tensors
    x = np.array([-128, 0, 127], dtype=np.int8)
    y = np.array([-128, 0, 127], dtype=np.int8)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: int16 tensors
    x = np.array([32767, 16384, 0], dtype=np.int16)
    y = np.array([32767, 16384, 0], dtype=np.int16)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: float32 tensors with broadcasting
    x = np.array([5.0, 4.0, 6.0], dtype=np.float32)
    y = np.array([5.0], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: half tensors
    x = np.array([1.5, 2.7, 3.1], dtype=np.float16)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {
        "name": "test9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: uint16 tensors - This will be removed due to invalid dtype
    # x = np.array([65535, 32768, 0], dtype=np.uint16)
    # y = np.array([32768, 16384, 0], dtype=np.uint16)
    # input_dict = {
    #     "name": "test10",
    #     "x": x,
    #     "y": y
    # }
    # list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Greater"] = tf_greater_inputs()

import numpy as np
import tensorflow as tf

def generate_l2loss_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with positive values
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "test1", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 2: 2D tensor with negative values
    t = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "test2", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D tensor with mixed values
    t = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {"name": "test3", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 4: 3D tensor with positive values
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"name": "test4", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 5: 3D tensor with negative values
    t = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "test5", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 6: 1D tensor with zero values
    t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"name": "test6", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 7: 2D tensor with float64 values
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"name": "test7", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 8: 2D tensor with float64 values and negative values
    t = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    input_dict = {"name": "test8", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 9: 1D tensor with bfloat16 values
    t = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"name": "test9", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 10: 2D tensor with half values
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"name": "test10", "t": t}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = generate_l2loss_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_log_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    input_dict = {
        'name': 'test1',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float64)
    input_dict = {
        'name': 'test2',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64 tensor with negative values
    x = np.array([-1.0+0j, -0.5+0j, 0.5+0j, 1.0+0j], dtype=np.complex64)
    input_dict = {
        'name': 'test3',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128 tensor with negative values
    x = np.array([-1.0+0j, -0.5+0j, 0.5+0j, 1.0+0j], dtype=np.complex128)
    input_dict = {
        'name': 'test4',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 tensor with zero values
    x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        'name': 'test5',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor with zero values
    x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    input_dict = {
        'name': 'test6',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: complex64 tensor with zero values
    x = np.array([0.0+0j, 0.5+0j, 1.0+0j, 2.0+0j], dtype=np.complex64)
    input_dict = {
        'name': 'test7',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: complex128 tensor with zero values
    x = np.array([0.0+0j, 0.5+0j, 1.0+0j, 2.0+0j], dtype=np.complex128)
    input_dict = {
        'name': 'test8',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float16)
    input_dict = {
        'name': 'test9',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: bfloat16 tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    input_dict = {
        'name': 'test10',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_maximum_inputs():
    list_of_inputs = []
    
    # Input 1: Basic float32 tensors
    x = np.array([0., 0., 0., 0.], dtype=np.float32)
    y = np.array([-2., 0., 2., 5.], dtype=np.float32)
    input_dict = {
        "name": "maximum_1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Float64 tensors
    x = np.array([1.5, 2.7, 3.9], dtype=np.float64)
    y = np.array([1.2, 2.8, 3.1], dtype=np.float64)
    input_dict = {
        "name": "maximum_2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Int32 tensors with negative values
    x = np.array([-1, -5, 3], dtype=np.int32)
    y = np.array([0, -2, 4], dtype=np.int32)
    input_dict = {
        "name": "maximum_3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Int64 tensors with broadcasting
    x = np.array([-5., 0., 0., 0.], dtype=np.float64)
    y = np.array([-3.], dtype=np.float64)
    input_dict = {
        "name": "maximum_4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Uint8 tensors
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([0, 4, 5], dtype=np.uint8)
    input_dict = {
        "name": "maximum_5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float32 with negative values and different dimensions
    x = np.array([[-1., 2., -3.], [4., -5., 6.]], dtype=np.float32)
    y = np.array([0., -1., 2.], dtype=np.float32)
    input_dict = {
        "name": "maximum_6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Int16 tensors with mixed positive and negative values
    x = np.array([1, -2, 3], dtype=np.int16)
    y = np.array([-1, 2, -3], dtype=np.int16)
    input_dict = {
        "name": "maximum_7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Half tensors
    x = np.array([1.5, 2.7], dtype=np.float16)
    y = np.array([1.2, 2.8], dtype=np.float16)
    input_dict = {
        "name": "maximum_8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Bfloat16 tensors
    x = np.array([1.5, 2.7], dtype=np.float32)
    y = np.array([1.2, 2.8], dtype=np.float32)
    input_dict = {
        "name": "maximum_9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex broadcasting with different shapes
    x = np.array([[-5., 0., 0., 0.], [-3., 1., 2., 3.]], dtype=np.float64)
    y = np.array([-3.], dtype=np.float64)
    input_dict = {
        "name": "maximum_10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Maximum"] = tf_maximum_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Mean_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "mean_1"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "mean_2"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "mean_3"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    axis_tensor = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "mean_4"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    axis_tensor = np.array([2], dtype=np.int64)
    keep_dims = True
    name = "mean_5"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "mean_6"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    axis_tensor = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "mean_7"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "mean_8"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.complex64)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "mean_9"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "mean_10"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_Mean_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_pow_inputs():
    list_of_inputs = []
    
    # Input 1: Float32 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.float32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float32)
    input_dict = {
        "name": "pow_1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Int32 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {
        "name": "pow_2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Float64 tensors
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {
        "name": "pow_3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    input_dict = {
        "name": "pow_4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Int64 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    input_dict = {
        "name": "pow_5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative values
    x = np.array([[-2, -3], [-4, -5]], dtype=np.float32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float32)
    input_dict = {
        "name": "pow_6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single dimension tensor
    x = np.array([2, 3, 4], dtype=np.float32)
    y = np.array([2, 3, 4], dtype=np.float32)
    input_dict = {
        "name": "pow_7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Scalar tensors
    x = np.array(2, dtype=np.float32)
    y = np.array(3, dtype=np.float32)
    input_dict = {
        "name": "pow_8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Float16 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.float16)
    y = np.array([[2, 3], [4, 5]], dtype=np.float16)
    input_dict = {
        "name": "pow_9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Half tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.half)
    y = np.array([[2, 3], [4, 5]], dtype=np.half)
    input_dict = {
        "name": "pow_10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Prod_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 2D tensor with axis=0
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "test1"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with axis=1
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis_tensor = np.array([1], dtype=np.int64)
    keep_dims = False
    name = "test2"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor with axis=2, keep_dims=True
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int32)
    axis_tensor = np.array([2], dtype=np.int32)
    keep_dims = True
    name = "test3"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D tensor with axis=0
    input_tensor = np.array([1, 2, 3, 4], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test4"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative values in tensor with axis=0
    input_tensor = np.array([[-1, -2, 3], [4, -5, 6]], dtype=np.int16)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "test5"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single element tensor with axis=0
    input_tensor = np.array([[[[1]]]], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test6"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Mixed tensor with axis=1, keep_dims=True
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "test7"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex tensor with axis=0
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test8"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Tensor with negative axis (e.g., -1)
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis_tensor = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "test9"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large values in tensor with axis=1 (use int32 instead of uint16)
    input_tensor = np.array([[[100, 200], [300, 400]], [[500, 600], [700, 800]]], dtype=np.int32)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "test10"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Prod"] = tf_raw_ops_Prod_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_real_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = np.array([-1-2j, -3-4j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = np.array([[-1-2j, -3-4j], [-5-6j, -7-8j]], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = np.array([0+0j, 1+0j, 0+1j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = np.array([1.5+2.5j, 3.5+4.5j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = np.array([1+2j, 3+4j, 5+6j, 7+8j, 9+10j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j], [9+10j, 11+12j]], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = np.array([-1-2j, -3-4j, -5-6j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = np.array([0+1j, 0+2j, 0+3j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_raw_ops_real_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_round_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor
    x = np.array([1.4, 2.7, 3.1, 4.5], dtype=np.float32)
    input_dict = {
        "name": "round1",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor
    x = np.array([1.4, 2.7, 3.1, 4.5], dtype=np.float64)
    input_dict = {
        "name": "round2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: int32 tensor
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {
        "name": "round3",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: int64 tensor
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict = {
        "name": "round4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 tensor with negative values
    x = np.array([-1.4, -2.7, -3.1, -4.5], dtype=np.float32)
    input_dict = {
        "name": "round5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor with negative values
    x = np.array([-1.4, -2.7, -3.1, -4.5], dtype=np.float64)
    input_dict = {
        "name": "round6",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: int32 tensor with negative values
    x = np.array([-1, -2, -3, -4], dtype=np.int32)
    input_dict = {
        "name": "round7",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: int64 tensor with negative values
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    input_dict = {
        "name": "round8",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with mixed values
    x = np.array([1.5, -2.5, 3.5, -4.5], dtype=np.float32)
    input_dict = {
        "name": "round9",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 tensor with mixed values
    x = np.array([1.5, -2.5, 3.5, -4.5], dtype=np.float64)
    input_dict = {
        "name": "round10",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Round"] = generate_round_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_selu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    features = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_1",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    features = np.array([[-1., -2., -3.], [4., 5., 6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_2",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    features = np.array([[-1.], [-2.], [-3.]], dtype=np.float32)
    input_dict = {
        "name": "selu_3",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    features = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]], dtype=np.float32)
    input_dict = {
        "name": "selu_4",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    features = np.array([[-1., -2., -3., -4., -5.]], dtype=np.float32)
    input_dict = {
        "name": "selu_5",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    features = np.array([[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]], dtype=np.float32)
    input_dict = {
        "name": "selu_6",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    features = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]], dtype=np.float32)
    input_dict = {
        "name": "selu_7",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    features = np.array([[[-1., -2., -3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    input_dict = {
        "name": "selu_8",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    features = np.array([[[[-1., -2., -3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    input_dict = {
        "name": "selu_9",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    features = np.array([[-1., -2., -3., -4., -5., -6., -7., -8., -9., -10.]], dtype=np.float32)
    input_dict = {
        "name": "selu_10",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Selu"] = generate_selu_inputs()

import numpy as np
import tensorflow as tf

def sinh_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: float64 tensor with mixed values
    x = np.array([-np.inf, -9.5, -0.5, 1.5, 1.2, 2.7, 10.3, np.inf], dtype=np.float64)
    input_dict = {
        "name": "test2",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: half tensor with positive values
    x = np.array([0.5, 1.2, 2.7, 10.3], dtype=np.float16)
    input_dict = {
        "name": "test3",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: bfloat16 tensor with negative values
    x = np.array([-1.5, -2.7, -10.3], dtype=np.float32)
    input_dict = {
        "name": "test4",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: complex64 tensor with real part
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {
        "name": "test5",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: complex128 tensor with real part
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    input_dict = {
        "name": "test6",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: scalar tensor (0D)
    x = np.array(5.5, dtype=np.float32)
    input_dict = {
        "name": "test7",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: 1D tensor with multiple negative values
    x = np.array([-1.5, -2.7, -10.3, -100], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: 3D tensor with multiple dimensions
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "name": "test9",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: tensor with zero values
    x = np.array([0.0, -0.5, 1.5], dtype=np.float32)
    input_dict = {
        "name": "test10",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Sinh"] = sinh_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_softplus_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_1",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_2",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    features = np.array([1.0], dtype=np.float64)
    input_dict = {
        "name": "softplus_3",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    features = np.array([-1.0], dtype=np.float64)
    input_dict = {
        "name": "softplus_4",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_5",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    features = np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_6",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    features = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_7",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_8",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    features = np.array([[-1.0], [-2.0], [-3.0], [-4.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_9",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    features = np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_10",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = generate_softplus_inputs()

import numpy as np
import tensorflow as tf

def generate_sparse_segment_mean_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with float32 data
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test1",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: With negative values
    data = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test2",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: With float64 data
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test3",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: With half data
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test4",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: With bfloat16 data
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test5",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Different segment IDs (not consecutive)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": True,
        "name": "test6",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: With int64 indices and segment IDs
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    segment_ids = np.array([0, 0], dtype=np.int64)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test7",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Mixed dimensions (2D data, 1D indices and segment_ids)
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test8",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: Large values in data
    data = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test9",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: Non-consecutive segment IDs with sorted values
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test10",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentMean"] = generate_sparse_segment_mean_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_sparse_segment_sum_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with float32 data
    data = np.array([[1., 2., 3., 4.], [-1., -2., -3., -4.], [5., 6., 7., 8.]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Different segments
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: All rows with two segments
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1, 2])
    segment_ids = np.array([0, 0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Float64 data
    data = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Int32 data
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative values in data
    data = np.array([[-1, -2, -3, -4], [1, 2, 3, 4]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Repeated segment ids
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1, 2])
    segment_ids = np.array([0, 0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: High dimensional tensor
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Mixed types (float32 and int32)
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1])
    segment_ids = np.array([0, 0])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": True,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different number of dimensions for indices and segment_ids
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    indices = np.array([0, 1, 2])
    segment_ids = np.array([0, 0, 1])
    
    input_dict = {
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids,
        "sparse_gradient": False,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSum"] = generate_sparse_segment_sum_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_tan_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float32)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64 tensor with real and imaginary parts
    x = np.array([complex(-0.5, 1.2), complex(1, 2), complex(3, 4)], dtype=np.complex64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128 tensor with real and imaginary parts
    x = np.array([complex(-0.5, 1.2), complex(1, 2), complex(3, 4)], dtype=np.complex128)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: half tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float16)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float32 scalar tensor
    x = np.array(1.5, dtype=np.float32)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float64 scalar tensor
    x = np.array(1.5, dtype=np.float64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: half scalar tensor
    x = np.array(1.5, dtype=np.float16)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: complex64 scalar tensor
    x = np.array(complex(1.5, 2.5), dtype=np.complex64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: complex128 scalar tensor
    x = np.array(complex(1.5, 2.5), dtype=np.complex128)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Tan"] = generate_tan_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_unicode_script_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D array
    input_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {
        "input": input_array,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array
    input_array_2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "input": input_array_2d,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Single element
    input_array_single = np.array([100], dtype=np.int32)
    input_dict = {
        "input": input_array_single,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values
    input_array_neg = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {
        "input": input_array_neg,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Mixed values including negative
    input_array_mix = np.array([1, -2, 3], dtype=np.int32)
    input_dict = {
        "input": input_array_mix,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values
    input_array_large = np.array([1000, 2000, 3000], dtype=np.int32)
    input_dict = {
        "input": input_array_large,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D array
    input_array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "input": input_array_3d,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single element in 1D array
    input_array_single_dim = np.array([0], dtype=np.int32)
    input_dict = {
        "input": input_array_single_dim,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Empty array
    input_array_empty = np.array([], dtype=np.int32)
    input_dict = {
        "input": input_array_empty,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex mix of values
    input_array_complex = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    input_dict = {
        "input": input_array_complex,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_unicode_script_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with True values
    condition = np.array([[True, False], [True, False]], dtype=bool)
    input_dict = {
        "name": "test1",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with multiple True values
    condition = np.array([[[True, False], [True, False]], [[False, True], [False, True]], [[False, False], [False, True]]], dtype=bool)
    input_dict = {
        "name": "test2",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor with all True values
    condition = np.array([[[True, True], [True, True]], [[True, True], [True, True]]], dtype=bool)
    input_dict = {
        "name": "test3",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with False values
    condition = np.array([[False, False], [False, False]], dtype=bool)
    input_dict = {
        "name": "test4",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with True values
    condition = np.array([True, False, True], dtype=bool)
    input_dict = {
        "name": "test5",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor with mixed values
    condition = np.array([[[1.5, 0.0], [-0.5, 0.0]], [[0.0, 0.25], [0.0, 0.75]], [[0.0, 0.0], [0.0, 0.01]]], dtype=np.float32)
    input_dict = {
        "name": "test6",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with complex values
    condition = np.array([[[1.5+0.0j, 0.0+0.0j], [0.0+0.5j, 0.0+0.0j]], [[0.0+0.0j, 0.25+1.5j], [0.0+0.0j, 0.75+0.0j]], [[0.0+0.0j, 0.0+0.0j], [0.0+0.0j, 0.01+0.0j]]], dtype=np.complex64)
    input_dict = {
        "name": "test7",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with negative values
    condition = np.array([[1.5, -0.5], [-0.5, 1.5]], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with all zeros
    condition = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.float32)
    input_dict = {
        "name": "test9",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D tensor with multiple True values
    condition = np.array([True, False, True, False, True], dtype=bool)
    input_dict = {
        "name": "test10",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_where_inputs()

import numpy as np
import tensorflow as tf

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: Basic string content
    filename = np.array("test_file.txt", dtype=np.string_)
    contents = np.array("Hello World!", dtype=np.string_)
    input_dict = {
        "name": "test_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Empty string content
    filename = np.array("empty_file.txt", dtype=np.string_)
    contents = np.array("", dtype=np.string_)
    input_dict = {
        "name": "empty_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: Multi-line string content
    filename = np.array("multiline_file.txt", dtype=np.string_)
    contents = np.array("Line 1\nLine 2\nLine 3", dtype=np.string_)
    input_dict = {
        "name": "multiline_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: String with special characters
    filename = np.array("special_chars.txt", dtype=np.string_)
    contents = np.array("Hello!\n@#$%^&*()", dtype=np.string_)
    input_dict = {
        "name": "special_chars.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: Numeric string content
    filename = np.array("numeric_file.txt", dtype=np.string_)
    contents = np.array("123456789", dtype=np.string_)
    input_dict = {
        "name": "numeric_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Binary content
    filename = np.array("binary_file.bin", dtype=np.string_)
    contents = np.array(b"\x00\x01\x02\x03", dtype=np.string_)
    input_dict = {
        "name": "binary_file.bin",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Long string content
    filename = np.array("long_file.txt", dtype=np.string_)
    contents = np.array("A" * 1000, dtype=np.string_)
    input_dict = {
        "name": "long_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: String with spaces and tabs
    filename = np.array("spaces_file.txt", dtype=np.string_)
    contents = np.array("   \t\t\t   ", dtype=np.string_)
    input_dict = {
        "name": "spaces_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: String with newline and carriage return
    filename = np.array("cr_newline_file.txt", dtype=np.string_)
    contents = np.array("\r\nHello\nWorld\r", dtype=np.string_)
    input_dict = {
        "name": "cr_newline_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: String with ASCII characters only
    filename = np.array("ascii_file.txt", dtype=np.string_)
    contents = np.array("Hello World! This is a test.", dtype=np.string_)
    input_dict = {
        "name": "ascii_file.txt",
        "filename": filename,
        "contents": contents
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.WriteFile"] = generate_inputs()

import numpy as np
import tensorflow as tf

def tf_sparse_SparseTensor_inputs():
    list_of_inputs = []
    
    # Input 1: Basic 2D sparse tensor
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: 3D sparse tensor with negative values
    indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    values = np.array([-1.5, 2.7], dtype=np.float32)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D sparse tensor with single element
    indices = np.array([[5]], dtype=np.int64)
    values = np.array([100], dtype=np.int64)
    dense_shape = np.array([10], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: 4D sparse tensor with mixed types
    indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64)
    values = np.array([1e-5, 2.5], dtype=np.float64)
    dense_shape = np.array([2, 2, 2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: Sparse tensor with repeated indices (not strictly required but valid)
    indices = np.array([[0, 1], [0, 1]], dtype=np.int64)
    values = np.array([3, 4], dtype=np.int64)
    dense_shape = np.array([3, 3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Sparse tensor with zero values (not strictly required but valid)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([0, 0], dtype=np.float32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Sparse tensor with float values and negative indices (not valid for sparse tensor but included for variety)
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1.5, -2.7], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: Sparse tensor with different data types
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([100, 200], dtype=np.int32)
    dense_shape = np.array([2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: Sparse tensor with multiple dimensions and large values
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values = np.array([1000, 2000], dtype=np.float32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Sparse tensor with complex shape and multiple elements
    indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    values = np.array([5.5, 6.7, 7.9], dtype=np.float32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor"] = tf_sparse_SparseTensor_inputs()

import numpy as np
import tensorflow as tf

def tf_train_coordinator_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.InvalidArgumentError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 2, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.NotFoundError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 3, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.UnavailableError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 4, valid
    input_dict = {
        "clean_stop_exception_types": ()
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.OutOfRangeError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 6, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.InternalError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 7, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.ResourceExhaustedError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 8, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.DeadlineExceededError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 9, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.PermissionDeniedError,)
    }
    list_of_inputs.append(input_dict)
    
    # Input 10, valid
    input_dict = {
        "clean_stop_exception_types": (tf.errors.AbortedError,)
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.train.Coordinator"] = tf_train_coordinator_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_convert_to_tensor_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test1"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    value = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test2"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    value = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    dtype = tf.float64
    dtype_hint = tf.float64
    name = "test3"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test4"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    dtype = tf.int64
    dtype_hint = tf.int64
    name = "test5"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    value = np.array([1.0, 2.0, None, 4.0], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test6"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    value = np.array([1, 2, 3], dtype=np.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test7"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = tf.float32
    dtype_hint = tf.float32
    name = "test8"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    value = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dtype = tf.float64
    dtype_hint = tf.float64
    name = "test9"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    dtype = tf.int32
    dtype_hint = tf.int32
    name = "test10"
    
    input_dict = {
        "value": value,
        "dtype": dtype,
        "dtype_hint": dtype_hint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.convert_to_tensor"] = tf_convert_to_tensor_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_dot_inputs():
    list_of_inputs = []
    
    # Input 1: Valid 2D tensors
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Valid 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Valid 3D tensors
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Mixed types (float and int)
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5, 6], [7, 8]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative values
    a = np.array([[-1, -2], [-3, -4]])
    b = np.array([[1, 2], [3, 4]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Scalar tensors (1D)
    a = np.array([1])
    b = np.array([2])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different dimensions
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4], [5, 6]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Zero values
    a = np.array([[0, 0], [0, 0]])
    b = np.array([[1, 2], [3, 4]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Floating point tensors
    a = np.array([[1.5, 2.7], [3.1, 4.8]])
    b = np.array([[5.2, 6.3], [7.9, 8.4]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large values (int)
    a = np.array([[100, 200], [300, 400]])
    b = np.array([[500, 600], [700, 800]])
    
    input_dict = {
        "a": a,
        "b": b
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.dot"] = tf_dot_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_isfinite_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([np.inf, -np.inf, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([np.nan, 1.0, 2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1.0, np.nan, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[np.inf, 2.0], [3.0, -np.inf]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[[np.nan, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-np.inf, np.inf, 0.0], [1.0, 2.0, 3.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isfinite"] = tf_isfinite_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_resize_with_crop_or_pad_inputs():
    list_of_inputs = []
    
    # Input 1: 3D image, target smaller than image
    image = np.arange(27).reshape(3, 3, 3)
    target_height = 2
    target_width = 2
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D image, target larger than image
    image = np.arange(18).reshape(3, 3, 2)
    target_height = 5
    target_width = 5
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D image, target smaller than image
    image = np.arange(108).reshape(4, 3, 3, 3)
    target_height = 2
    target_width = 2
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D image, target larger than image
    image = np.arange(36).reshape(1, 3, 3, 4)
    target_height = 5
    target_width = 5
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D image, target same size as image
    image = np.arange(27).reshape(3, 3, 3)
    target_height = 3
    target_width = 3
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D image, target larger than image with zero padding
    image = np.arange(18).reshape(3, 3, 2)
    target_height = 4
    target_width = 4
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D image, target smaller than image with cropping
    image = np.arange(27).reshape(3, 3, 3)
    target_height = 2
    target_width = 2
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D image, target larger than image with padding
    image = np.arange(108).reshape(4, 3, 3, 3)
    target_height = 5
    target_width = 5
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D image, target smaller than image with cropping
    image = np.arange(108).reshape(4, 3, 3, 3)
    target_height = 2
    target_width = 2
    
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.resize_with_crop_or_pad"] = tf_image_resize_with_crop_or_pad_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linear_operator_householder_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic Householder reflection with 1D vector
    reflection_axis = np.array([1.0, 0.0])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_1D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - 2D vector with non-singular flag
    reflection_axis = np.array([0.5, 0.5])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_2D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 3D vector with positive definite flag
    reflection_axis = np.array([1.0, 1.0, 1.0])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_3D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 4D vector with square flag
    reflection_axis = np.array([1.0, 0.0, 0.0, 1.0])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_4D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - Batch vector with non-singular flag
    reflection_axis = np.array([[1.0, 0.0], [0.0, 1.0]])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_batch"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Negative values vector
    reflection_axis = np.array([-1.0, 1.0])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_negative"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Floating point vector with non-singular flag
    reflection_axis = np.array([0.1, 0.2])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_float"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Zero vector (not recommended but valid)
    reflection_axis = np.array([0.0, 0.0])
    is_non_singular = False
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_zero"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Higher dimensional vector (5D)
    reflection_axis = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "Householder_5D"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Complex vector with multiple dimensions
    reflection_axis = np.array([1.0 + 1j, 0.0 + 0j])
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "Householder_complex"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorHouseholder"] = tf_linalg_linear_operator_householder_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_solve_inputs():
    list_of_inputs = []
    
    # Input 1, valid - basic case with float32
    matrix = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    adjoint = False
    name = "test"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - with complex64
    matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    rhs = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    adjoint = True
    name = "test2"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - with float64
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    adjoint = False
    name = "test3"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - with complex128
    matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    rhs = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    adjoint = True
    name = "test4"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - negative values with float32
    matrix = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0], [-7.0, 8.0]], dtype=np.float32)
    adjoint = True
    name = "test5"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - with negative values
    matrix = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, -6.0], [7.0, -8.0]], dtype=np.float32)
    adjoint = True
    name = "test6"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - single dimension matrices with float32
    matrix = np.array([[1.0]], dtype=np.float32)
    rhs = np.array([[2.0]], dtype=np.float32)
    adjoint = False
    name = "test7"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - float32 with more complex structure but invertible
    matrix = np.array([[2.0, 1.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    adjoint = False
    name = "test8"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - with float32 but different dimensions
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    adjoint = False
    name = "test9"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - float32 with different dimensions and invertible matrix
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    adjoint = True
    name = "test10"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.solve"] = tf_linalg_solve_inputs()

import numpy as np
import tensorflow as tf

def generate_draw_bounding_boxes_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 4D image tensor and 3D boxes tensor
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    boxes = np.random.rand(2, 1, 4).astype(np.float32)
    input_dict = {
        "name": "test1",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Different batch size
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(1, 2, 4).astype(np.float32)
    input_dict = {
        "name": "test2",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: Different depth
    images = np.random.rand(1, 64, 64, 1).astype(np.float32)
    boxes = np.random.rand(1, 3, 4).astype(np.float32)
    input_dict = {
        "name": "test3",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: Multiple bounding boxes per image
    images = np.random.rand(2, 128, 128, 3).astype(np.float32)
    boxes = np.random.rand(2, 5, 4).astype(np.float32)
    input_dict = {
        "name": "test4",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: With negative values in boxes (valid since it's a float tensor)
    images = np.random.rand(2, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(2, 1, 4).astype(np.float32) * -1
    input_dict = {
        "name": "test5",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Non-square image (height != width)
    images = np.random.rand(2, 80, 120, 3).astype(np.float32)
    boxes = np.random.rand(2, 1, 4).astype(np.float32)
    input_dict = {
        "name": "test6",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Different coordinate format (e.g. [y_min, x_min, y_max, x_max])
    images = np.random.rand(3, 40, 40, 3).astype(np.float32)
    boxes = np.random.rand(3, 1, 4).astype(np.float32)
    input_dict = {
        "name": "test7",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: Very large number of bounding boxes (e.g. 50)
    images = np.random.rand(1, 64, 64, 3).astype(np.float32)
    boxes = np.random.rand(1, 50, 4).astype(np.float32)
    input_dict = {
        "name": "test8",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: Very small bounding box coordinates (close to zero)
    images = np.random.rand(1, 32, 32, 3).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32) * 0.01
    input_dict = {
        "name": "test9",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Large bounding box coordinates (close to 1.0)
    images = np.random.rand(1, 64, 64, 3).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32) * 0.99
    input_dict = {
        "name": "test10",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = generate_draw_bounding_boxes_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_shuffle_inputs():
    list_of_inputs = []
    
    # Input 1
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {
        "value": value,
        "seed": 1,
        "seed2": 2,
        "name": "shuffle_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "value": value,
        "seed": 3,
        "seed2": 4,
        "name": "shuffle_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    value = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    input_dict = {
        "value": value,
        "seed": 5,
        "seed2": 6,
        "name": "shuffle_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    value = np.array([[[1, 2]], [[3, 4]]], dtype=np.float64)
    input_dict = {
        "value": value,
        "seed": 7,
        "seed2": 8,
        "name": "shuffle_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    value = np.array([[-1, -2, -3], [1, 2, 3]], dtype=np.int32)
    input_dict = {
        "value": value,
        "seed": 9,
        "seed2": 10,
        "name": "shuffle_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    value = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    input_dict = {
        "value": value,
        "seed": 11,
        "seed2": 12,
        "name": "shuffle_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "value": value,
        "seed": 13,
        "seed2": 14,
        "name": "shuffle_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int64)
    input_dict = {
        "value": value,
        "seed": 15,
        "seed2": 16,
        "name": "shuffle_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], dtype=np.float64)
    input_dict = {
        "value": value,
        "seed": 17,
        "seed2": 18,
        "name": "shuffle_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    value = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    input_dict = {
        "value": value,
        "seed": 19,
        "seed2": 20,
        "name": "shuffle_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffle"] = tf_random_shuffle_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []
    
    # Input 1: 4D tensor with axis [3]
    tensor_1 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_1 = np.array([3], dtype=np.int32)
    input_dict_1 = {
        "tensor": tensor_1,
        "axis": axis_1,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    # Input 2: 4D tensor with axis [-1]
    tensor_2 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_2 = np.array([-1], dtype=np.int32)
    input_dict_2 = {
        "tensor": tensor_2,
        "axis": axis_2,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    # Input 3: 4D tensor with axis [1]
    tensor_3 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_3 = np.array([1], dtype=np.int32)
    input_dict_3 = {
        "tensor": tensor_3,
        "axis": axis_3,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: 4D tensor with axis [-3]
    tensor_4 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_4 = np.array([-3], dtype=np.int32)
    input_dict_4 = {
        "tensor": tensor_4,
        "axis": axis_4,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: 4D tensor with axis [2]
    tensor_5 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_5 = np.array([2], dtype=np.int32)
    input_dict_5 = {
        "tensor": tensor_5,
        "axis": axis_5,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: 4D tensor with axis [-2]
    tensor_6 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_6 = np.array([-2], dtype=np.int32)
    input_dict_6 = {
        "tensor": tensor_6,
        "axis": axis_6,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 3D tensor with axis [0]
    tensor_7 = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=np.int32)
    axis_7 = np.array([0], dtype=np.int32)
    input_dict_7 = {
        "tensor": tensor_7,
        "axis": axis_7,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: 2D tensor with axis [1]
    tensor_8 = np.array([[0, 1], [2, 3]], dtype=np.int32)
    axis_8 = np.array([1], dtype=np.int32)
    input_dict_8 = {
        "tensor": tensor_8,
        "axis": axis_8,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: 4D tensor with axis [0, 1]
    tensor_9 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_9 = np.array([0, 1], dtype=np.int32)
    input_dict_9 = {
        "tensor": tensor_9,
        "axis": axis_9,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 4D tensor with axis [1, 3]
    tensor_10 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_10 = np.array([1, 3], dtype=np.int32)
    input_dict_10 = {
        "tensor": tensor_10,
        "axis": axis_10,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

import tensorflow as tf
import copy
import numpy as np

def tf_sets_union_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[2, 4, -6], [5, 7, 9]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    validate_indices = False

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 2]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 2]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 2]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 2]
    )
    validate_indices = False

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - different sizes
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 3]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 2]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - with negative values
    a = tf.constant([[1, 2, -3], [4, 5, 6]])
    b = tf.constant([[2, 4, -6], [5, 7, 9]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid - higher dimensional
    a = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    b = tf.constant([[[[2, 3], [4, 5]], [[6, 7], [8, 9]]], [[[10, 11], [12, 13]], [[14, 15], [16, 17]]]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid - same shape but different values
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[7, 8, 9], [10, 11, 12]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid - sparse with different shapes
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 3]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
        values=[2, 4, 5, 6],
        dense_shape=[2, 4]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid - sparse with overlapping indices
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 3]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 3]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sets.union"] = tf_sets_union_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_grayscale_to_rgb_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D grayscale image
    images = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D grayscale image
    images = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D grayscale image with multiple channels
    images = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D grayscale image
    images = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative values in grayscale image
    images = np.array([[[1.0], [-2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values in grayscale image
    images = np.array([[[100.0], [200.0], [300.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float values in grayscale image
    images = np.array([[[1.5], [2.7], [3.9]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Mixed values in grayscale image
    images = np.array([[[1.0], [2.5], [3.9]], [[4.2], [5.7], [6.1]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Single pixel grayscale image
    images = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Multiple grayscale images with different shapes
    images = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=np.float32)
    input_dict = {
        "images": images,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.grayscale_to_rgb"] = generate_grayscale_to_rgb_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor (height, width, channels)
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed = np.array([2, 3], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor (batch, height, width, channels) - Corrected format
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor with different channel count
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    seed = np.array([9, 0], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with float values
    image = np.array([[[1.5], [2.5]], [[3.5], [4.5]]], dtype=np.float32)
    seed = np.array([5, 6], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with negative values
    image = np.array([[[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]], dtype=np.int32)
    seed = np.array([-1, -2], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with mixed values
    image = np.array([[[[1, -2], [3, -4]], [[5, -6], [7, -8]]]], dtype=np.int32)
    seed = np.array([10, 11], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with zero values
    image = np.array([[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]], dtype=np.int32)
    seed = np.array([12, 13], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with larger values
    image = np.array([[[[100, 200], [300, 400]], [[500, 600], [700, 800]]]], dtype=np.int32)
    seed = np.array([14, 15], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with single channel
    image = np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.int32)
    seed = np.array([16, 17], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with different channel counts
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    seed = np.array([18, 19], dtype=np.int32)
    
    input_dict = {
        "image": image,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_left_right"] = generate_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_inputs():
    list_of_inputs = []
    
    # Input 1: Scalar indices
    params = np.array([[1, 2, 3], [4, 5, 6]])
    indices = np.array(1, dtype=np.int32)
    validate_indices = True
    name = "test1"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Vector indices
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = np.array([0, 1, 2], dtype=np.int32)
    validate_indices = True
    name = "test2"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Higher rank indices (2D)
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    validate_indices = True
    name = "test3"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Empty indices
    params = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([], dtype=np.int32)
    validate_indices = True
    name = "test4"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single element indices
    params = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0], dtype=np.int32)
    validate_indices = True
    name = "test5"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large indices (with repeated values)
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    indices = np.array([2, 2, 0, 1], dtype=np.int32)
    validate_indices = True
    name = "test6"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D indices with single element
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = np.array([2], dtype=np.int32)
    validate_indices = True
    name = "test7"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Multi-dimensional indices with shape (2, 2)
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices = np.array([[1, 0], [0, 1]], dtype=np.int32)
    validate_indices = True
    name = "test8"
    
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": validate_indices,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_gather_inputs()


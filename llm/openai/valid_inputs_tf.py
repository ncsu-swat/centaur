generated_inputs = {}
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_compat_path_to_str_inputs():
    list_of_inputs = []
    path1 = tf.constant("C:\\XYZ\\tensorflow\\./.././tensorflow")
    path2 = tf.constant("C:\XYZ\tensorflow\./.././tensorflow")
    path3 = tf.constant("./corpus")
    path4 = tf.constant("./.././Corpus")
    path5 = tf.constant("./.././Corpus")
    path6 = tf.constant("./..////../")
    path7 = tf.constant("path/to/file")
    path8 = tf.constant("another_path")
    path9 = tf.constant("path/with/../relative/path")
    path10 = tf.constant("C:/XYZ/tensorflow/./.././tensorflow")
    path11 = tf.constant("C:\\XYZ\\tensorflow\\./.././tensorflow")
    path12 = tf.constant("Relative/Path/Test")

    input_dict1 = {"path": path1}
    input_dict2 = {"path": path2}
    input_dict3 = {"path": path3}
    input_dict4 = {"path": path4}
    input_dict5 = {"path": path5}
    input_dict6 = {"path": path6}
    input_dict7 = {"path": path7}
    input_dict8 = {"path": path8}
    input_dict9 = {"path": path9}
    input_dict10 = {"path": path10}
    input_dict11 = {"path": path11}
    input_dict12 = {"path": path12}

    list_of_inputs.append(copy.deepcopy(input_dict1))
    list_of_inputs.append(copy.deepcopy(input_dict2))
    list_of_inputs.append(copy.deepcopy(input_dict3))
    list_of_inputs.append(copy.deepcopy(input_dict4))
    list_of_inputs.append(copy.deepcopy(input_dict5))
    list_of_inputs.append(copy.deepcopy(input_dict6))
    list_of_inputs.append(copy.deepcopy(input_dict7))
    list_of_inputs.append(copy.deepcopy(input_dict8))
    list_of_inputs.append(copy.deepcopy(input_dict9))
    list_of_inputs.append(copy.deepcopy(input_dict10))
    list_of_inputs.append(copy.deepcopy(input_dict11))
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs["tf.compat.path_to_str"] = tf_compat_path_to_str_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_isfinite_inputs():
    list_of_inputs = []
    input_dict = {}

    x = tf.constant([1.0, 2.0, float('inf'), -float('inf'), float('nan')])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant([[1.0, 2.0], [float('inf'), -float('inf')]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isfinite"] = tf_experimental_numpy_isfinite_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_feature_column_categorical_column_with_identity_inputs():
    list_of_inputs = []

    input_dict1 = {
        "key": "video_id",
        "num_buckets": 1000,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        "key": "product_id",
        "num_buckets": 500,
        "default_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        "key": "user_id",
        "num_buckets": 2000,
        "default_value": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        "key": "item_id",
        "num_buckets": 100,
        "default_value": 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        "key": "category",
        "num_buckets": 50,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        "key": "ad_id",
        "num_buckets": 10000,
        "default_value": 1000
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        "key": "location_id",
        "num_buckets": 100,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input_dict8 = {
        "key": "query_id",
        "num_buckets": 5000,
        "default_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        "key": "campaign_id",
        "num_buckets": 200,
        "default_value": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        "key": "device_id",
        "num_buckets": 1000,
        "default_value": 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["tf.feature_column.categorical_column_with_identity"] = tf_feature_column_categorical_column_with_identity_inputs()

import tensorflow as tf
import copy

def sequence_categorical_column_with_hash_bucket_inputs():
    list_of_inputs = []

    key1 = "tokens1"
    hash_bucket_size1 = 100
    dtype1 = tf.string
    input_dict1 = {
        "key": key1,
        "hash_bucket_size": hash_bucket_size1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    key2 = "tokens2"
    hash_bucket_size2 = 500
    dtype2 = tf.int32
    input_dict2 = {
        "key": key2,
        "hash_bucket_size": hash_bucket_size2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    key3 = "tokens3"
    hash_bucket_size3 = 200
    dtype3 = tf.string
    input_dict3 = {
        "key": key3,
        "hash_bucket_size": hash_bucket_size3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    key4 = "tokens4"
    hash_bucket_size4 = 1000
    dtype4 = tf.int64
    input_dict4 = {
        "key": key4,
        "hash_bucket_size": hash_bucket_size4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    key5 = "tokens5"
    hash_bucket_size5 = 10
    dtype5 = tf.string
    input_dict5 = {
        "key": key5,
        "hash_bucket_size": hash_bucket_size5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    key6 = "tokens6"
    hash_bucket_size6 = 5
    dtype6 = tf.int32
    input_dict6 = {
        "key": key6,
        "hash_bucket_size": hash_bucket_size6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    key7 = "tokens7"
    hash_bucket_size7 = 250
    dtype7 = tf.string
    input_dict7 = {
        "key": key7,
        "hash_bucket_size": hash_bucket_size7,
        "dtype": dtype7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    key8 = "tokens8"
    hash_bucket_size8 = 150
    dtype8 = tf.int64
    input_dict8 = {
        "key": key8,
        "hash_bucket_size": hash_bucket_size8,
        "dtype": dtype8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    key9 = "tokens9"
    hash_bucket_size9 = 300
    dtype9 = tf.string
    input_dict9 = {
        "key": key9,
        "hash_bucket_size": hash_bucket_size9,
        "dtype": dtype9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    key10 = "tokens10"
    hash_bucket_size10 = 75
    dtype10 = tf.int32
    input_dict10 = {
        "key": key10,
        "hash_bucket_size": hash_bucket_size10,
        "dtype": dtype10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = sequence_categorical_column_with_hash_bucket_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_get_static_value_inputs():
    list_of_inputs = []

    input_dict = {
        "tensor": tf.constant(10),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([1, 2, 3]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[1, 2], [3, 4]]),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(-5),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([-1, -2, -3]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(0.5),
        "partial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([1.0, 2.5, -3.2]),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(10),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant(-10),
        "partial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.get_static_value"] = tf_get_static_value_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_identity_inputs():
    list_of_inputs = []
    input_dict = {}
    

    input_dict = {
        "input": tf.constant([1, 2, 3]),
        "name": "identity_tensor_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([[1, 2], [3, 4]]),
        "name": "identity_tensor_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "name": "identity_tensor_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([-1, -2, -3]),
        "name": "identity_tensor_negative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([0.1, 0.2, 0.3]),
        "name": "identity_tensor_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1, 0, 1, 0]),
        "name": "identity_tensor_bool"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1 + 1j, 2 + 2j]),
        "name": "identity_tensor_complex"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1, 2, 3], dtype=tf.int64),
        "name": "identity_tensor_int64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([1.0, 2.0, 3.0], dtype=tf.float64),
        "name": "identity_tensor_float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": tf.constant([True, False, True], dtype=tf.bool),
        "name": "identity_tensor_bool_type"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.identity"] = tf_identity_inputs()


import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_adjust_saturation_inputs():
    list_of_inputs = []
    image1 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float32)
    saturation_factor1 = 0.5
    input_dict1 = {"image": image1, "saturation_factor": saturation_factor1, "name": "adjust_sat1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = tf.constant([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]], dtype=tf.float32)
    saturation_factor2 = 2.0
    input_dict2 = {"image": image2, "saturation_factor": saturation_factor2, "name": "adjust_sat2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=tf.float32)
    saturation_factor3 = -1.0
    input_dict3 = {"image": image3, "saturation_factor": saturation_factor3, "name": "adjust_sat3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float64)
    saturation_factor4 = 1.5
    input_dict4 = {"image": image4, "saturation_factor": saturation_factor4, "name": "adjust_sat4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = tf.zeros((10, 10, 3), dtype=tf.float32)
    saturation_factor5 = 0.0
    input_dict5 = {"image": image5, "saturation_factor": saturation_factor5, "name": "adjust_sat5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = tf.ones((5, 5, 3), dtype=tf.float32)
    saturation_factor6 = 1.0
    input_dict6 = {"image": image6, "saturation_factor": saturation_factor6, "name": "adjust_sat6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = tf.random.uniform((2, 2, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    saturation_factor7 = 0.75
    input_dict7 = {"image": image7, "saturation_factor": saturation_factor7, "name": "adjust_sat7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)
    saturation_factor8 = 2.5
    input_dict8 = {"image": image8, "saturation_factor": saturation_factor8, "name": "adjust_sat8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float64)
    saturation_factor9 = -0.5
    input_dict9 = {"image": image9, "saturation_factor": saturation_factor9, "name": "adjust_sat9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float32)
    saturation_factor10 = 3.0
    input_dict10 = {"image": image10, "saturation_factor": saturation_factor10, "name": "adjust_sat10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.adjust_saturation"] = tf_image_adjust_saturation_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    input_dict = {
        "tensor": tf.constant(1),
        "name": "scalar_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([1, 2, 3]),
        "name": "vector_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[1, 2], [3, 4]]),
        "name": "matrix_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "name": "3d_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]),
        "name": "4d_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_atan2_inputs():
    list_of_inputs = []

    y = tf.constant([1.0, -1.0], dtype=tf.float32)
    x = tf.constant([1.0, 1.0], dtype=tf.float32)
    name = "atan2_example"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = tf.constant([2.0, -3.0, 4.0], dtype=tf.float64)
    x = tf.constant([1.0, -2.0, 3.0], dtype=tf.float64)
    name = "atan2_example2"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = tf.constant([[1.0, -1.0], [2.0, -3.0]], dtype=tf.float32)
    x = tf.constant([[1.0, 1.0], [-2.0, 2.0]], dtype=tf.float32)
    name = "atan2_example3"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = tf.constant([[[1.0, -1.0]], [[2.0, -3.0]]], dtype=tf.float32)
    x = tf.constant([[[1.0, 1.0]], [[-2.0, 2.0]]], dtype=tf.float32)
    name = "atan2_example4"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    x = tf.constant([1.0, -1.0, 0.0], dtype=tf.float32)
    name = "atan2_example5"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = tf.constant([1.0, -1.0, 2.], dtype=tf.float32)
    x = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
    name = "atan2_example6"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.atan2"] = tf_atan2_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_empty_inputs():
    list_of_inputs = []
    input_dict_1 = {
        "shape": tf.constant([2, 3], dtype=tf.int32),
        "dtype": tf.float32,
        "init": True,
        "name": "empty_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        "shape": tf.constant([5]),
        "dtype": tf.int64,
        "init": False,
        "name": "empty_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        "shape": tf.constant([1, 4, 2]),
        "dtype": tf.string,
        "init": True,
        "name": "empty_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {
        "shape": tf.constant([0, 0]),
        "dtype": tf.bool,
        "init": False,
        "name": "empty_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {
        "shape": tf.constant([2, 2, 3, 1]),
        "dtype": tf.float16,
        "init": True,
        "name": "empty_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_dict_6 = {
        "shape": tf.constant([10, 1]),
        "dtype": tf.complex64,
        "init": False,
        "name": "empty_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_dict_7 = {
        "shape": tf.constant([2]),
        "dtype": tf.int32,
        "init": True,
        "name": "empty_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_dict_8 = {
         "shape": tf.constant([3, 4, 5]),
        "dtype": tf.float32,
        "init": False,
        "name": "empty_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_dict_9 = {
        "shape": tf.constant([1, 5, 1, 2]),
        "dtype": tf.string,
        "init": True,
        "name": "empty_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_dict_10 = {
        "shape": tf.constant([2, 3, 0, 1]),
        "dtype": tf.bool,
        "init": False,
        "name": "empty_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    return list_of_inputs

generated_inputs["tf.raw_ops.Empty"] = tf_empty_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_greater_equal_inputs():
    list_of_inputs = []
    x = tf.constant([5, 4, 6, 7], dtype=tf.float32)
    y = tf.constant([5, 2, 5, 10], dtype=tf.float32)
    input_dict = {'name': 'greater_equal_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant([5, 4, 6, 7], dtype=tf.int32)
    y = tf.constant([5], dtype=tf.int32)
    input_dict = {'name': 'greater_equal_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant([1, 2, 3, 4], dtype=tf.uint8)
    y = tf.constant([2, 1, 4, 3], dtype=tf.uint8)
    input_dict = {'name': 'greater_equal_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GreaterEqual"] = tf_raw_ops_greater_equal_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_real_inputs():
    list_of_inputs = []

    input_1 = tf.constant([1.0 + 2.0j, 3.0 - 4.0j], dtype=tf.complex64)
    input_dict_1 = {'input': input_1, 'Tout': tf.float32, 'name': 'real_op_1'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = tf.constant([5.0 + 6.0j, 7.0 - 8.0j], dtype=tf.complex128)
    input_dict_2 = {'input': input_2, 'Tout': tf.float64, 'name': 'real_op_2'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = tf.constant([[-1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 6.0j, -7.0 - 8.0j]], dtype=tf.complex64)
    input_dict_3 = {'input': input_3, 'Tout': tf.float32, 'name': 'real_op_3'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = tf.constant([1.0 + 2.0j, 3.0 - 4.0j], dtype=tf.complex128)
    input_dict_4 = {'input': input_4, 'Tout': tf.float64, 'name': 'real_op_4'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = tf.constant([1.0 + 0.0j, 3.0 - 0.0j], dtype=tf.complex64)
    input_dict_5 = {'input': input_5, 'Tout': tf.float32, 'name': 'real_op_5'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_real_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_relu_inputs():
    list_of_inputs = []

    features_float32 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"name": "relu_1", "features": features_float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features_float64 = np.array([-2.5, 0.0, 1.5], dtype=np.float64)
    input_dict = {"name": "relu_2", "features": features_float64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
generated_inputs["tf.raw_ops.Relu"] = tf_raw_ops_relu_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_reduce_sum_sparse_inputs():
    list_of_inputs = []

    input_dict1 = {
        "input_indices": tf.constant([[0, 0], [1, 1]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0], dtype=tf.float32),
        "input_shape": tf.constant([2, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        "input_indices": tf.constant([[0, 0], [0, 1], [1, 0]], dtype=tf.int64),
        "input_values": tf.constant([3, 4, 5], dtype=tf.int32),
        "input_shape": tf.constant([2, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([1], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        "input_indices": tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0, 3.0], dtype=tf.float64),
        "input_shape": tf.constant([3, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0, 1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        "input_indices": tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.int64),
        "input_values": tf.constant([1, 2, 3], dtype=tf.int64),
        "input_shape": tf.constant([3, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([-1], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        "input_indices": tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
        "input_shape": tf.constant([3, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([0, -1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        "input_indices": tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0, 3.0], dtype=tf.float64),
        "input_shape": tf.constant([3, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        "input_indices": tf.constant([[0, 0], [0, 1]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0], dtype=tf.float32),
        "input_shape": tf.constant([2, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input_dict8 = {
        "input_indices": tf.constant([[0, 0], [1, 1]], dtype=tf.int64),
        "input_values": tf.constant([1.0, 2.0], dtype=tf.float32),
        "input_shape": tf.constant([2, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        "input_indices": tf.constant([[0, 0], [0, 1], [1, 0]], dtype=tf.int64),
        "input_values": tf.constant([3, 4, 5], dtype=tf.int32),
        "input_shape": tf.constant([2, 3], dtype=tf.int64),
        "reduction_axes": tf.constant([1], dtype=tf.int32),
        "keep_dims": False,
        "name": "SparseReduceSumSparse_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        "input_indices": tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
        "input_values": tf.constant([1, 2, 3], dtype=tf.int64),
        "input_shape": tf.constant([3, 2], dtype=tf.int64),
        "reduction_axes": tf.constant([0, 1], dtype=tf.int32),
        "keep_dims": True,
        "name": "SparseReduceSumSparse_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceSumSparse"] = tf_sparse_reduce_sum_sparse_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sysconfig_get_include_inputs():
    list_of_inputs = []
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.sysconfig.get_include"] = tf_sysconfig_get_include_inputs()


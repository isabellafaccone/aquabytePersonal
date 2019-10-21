node {
  name: "input_placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
        dim {
          size: 3
        }
      }
    }
  }
}
node {
  name: "cmap_placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "gt_hmap_placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
        dim {
          size: 9
        }
      }
    }
  }
}
node {
  name: "pooled_center_map/center_map/AvgPool"
  op: "AvgPool"
  input: "cmap_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 9
        i: 9
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 8
        i: 8
        i: 1
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\003\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.09975093603134155
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.09975093603134155
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv1/kernel"
  input: "sub_stages/sub_conv1/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv1/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv1/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv1/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv1/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv1/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv1/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv1/bias"
  input: "sub_stages/sub_conv1/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv1/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv1/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv1/Conv2D"
  op: "Conv2D"
  input: "input_placeholder"
  input: "sub_stages/sub_conv1/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv1/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv1/Conv2D"
  input: "sub_stages/sub_conv1/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv1/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.07216878235340118
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.07216878235340118
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv2/kernel"
  input: "sub_stages/sub_conv2/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv2/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv2/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv2/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv2/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv2/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv2/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv2/bias"
  input: "sub_stages/sub_conv2/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv2/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv2/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv2/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv1/Relu"
  input: "sub_stages/sub_conv2/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv2/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv2/Conv2D"
  input: "sub_stages/sub_conv2/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv2/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_pool1/MaxPool"
  op: "MaxPool"
  input: "sub_stages/sub_conv2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0589255653321743
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0589255653321743
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv3/kernel"
  input: "sub_stages/sub_conv3/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv3/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv3/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv3/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv3/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv3/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv3/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv3/bias"
  input: "sub_stages/sub_conv3/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv3/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv3/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv3/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_pool1/MaxPool"
  input: "sub_stages/sub_conv3/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv3/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv3/Conv2D"
  input: "sub_stages/sub_conv3/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv3/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.05103103816509247
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.05103103816509247
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv4/kernel"
  input: "sub_stages/sub_conv4/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv4/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv4/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv4/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv4/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv4/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv4/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv4/bias"
  input: "sub_stages/sub_conv4/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv4/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv4/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv4/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv3/Relu"
  input: "sub_stages/sub_conv4/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv4/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv4/Conv2D"
  input: "sub_stages/sub_conv4/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv4/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_pool2/MaxPool"
  op: "MaxPool"
  input: "sub_stages/sub_conv4/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0416666679084301
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0416666679084301
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv5/kernel"
  input: "sub_stages/sub_conv5/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv5/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv5/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv5/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv5/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv5/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv5/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv5/bias"
  input: "sub_stages/sub_conv5/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv5/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv5/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv5/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_pool2/MaxPool"
  input: "sub_stages/sub_conv5/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv5/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv5/Conv2D"
  input: "sub_stages/sub_conv5/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv5/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv5/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.03608439117670059
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.03608439117670059
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv6/kernel"
  input: "sub_stages/sub_conv6/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv6/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv6/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv6/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv6/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv6/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv6/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv6/bias"
  input: "sub_stages/sub_conv6/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv6/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv6/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv6/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv5/Relu"
  input: "sub_stages/sub_conv6/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv6/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv6/Conv2D"
  input: "sub_stages/sub_conv6/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv6/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv6/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.03608439117670059
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.03608439117670059
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv7/kernel"
  input: "sub_stages/sub_conv7/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv7/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv7/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv7/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv7/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv7/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv7/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv7/bias"
  input: "sub_stages/sub_conv7/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv7/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv7/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv7/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv6/Relu"
  input: "sub_stages/sub_conv7/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv7/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv7/Conv2D"
  input: "sub_stages/sub_conv7/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv7/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv7/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.03608439117670059
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.03608439117670059
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv8/kernel"
  input: "sub_stages/sub_conv8/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv8/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv8/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv8/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv8/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv8/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv8/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv8/bias"
  input: "sub_stages/sub_conv8/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv8/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv8/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv8/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv7/Relu"
  input: "sub_stages/sub_conv8/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv8/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv8/Conv2D"
  input: "sub_stages/sub_conv8/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv8/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv8/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_pool3/MaxPool"
  op: "MaxPool"
  input: "sub_stages/sub_conv8/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02946278266608715
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02946278266608715
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv9/kernel"
  input: "sub_stages/sub_conv9/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv9/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv9/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv9/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv9/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv9/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv9/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv9/bias"
  input: "sub_stages/sub_conv9/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv9/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv9/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv9/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_pool3/MaxPool"
  input: "sub_stages/sub_conv9/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv9/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv9/Conv2D"
  input: "sub_stages/sub_conv9/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv9/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv9/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv10/kernel"
  input: "sub_stages/sub_conv10/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv10/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv10/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv10/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv10/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv10/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv10/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv10/bias"
  input: "sub_stages/sub_conv10/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv10/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv10/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv10/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv9/Relu"
  input: "sub_stages/sub_conv10/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv10/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv10/Conv2D"
  input: "sub_stages/sub_conv10/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv10/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv10/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv11/kernel"
  input: "sub_stages/sub_conv11/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv11/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv11/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv11/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv11/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv11/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv11/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv11/bias"
  input: "sub_stages/sub_conv11/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv11/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv11/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv11/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv10/Relu"
  input: "sub_stages/sub_conv11/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv11/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv11/Conv2D"
  input: "sub_stages/sub_conv11/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv11/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv11/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv12/kernel"
  input: "sub_stages/sub_conv12/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv12/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv12/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv12/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv12/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv12/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv12/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv12/bias"
  input: "sub_stages/sub_conv12/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv12/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv12/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv12/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv11/Relu"
  input: "sub_stages/sub_conv12/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv12/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv12/Conv2D"
  input: "sub_stages/sub_conv12/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv12/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv12/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv13/kernel"
  input: "sub_stages/sub_conv13/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv13/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv13/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv13/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv13/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv13/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv13/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv13/bias"
  input: "sub_stages/sub_conv13/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv13/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv13/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv13/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv12/Relu"
  input: "sub_stages/sub_conv13/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv13/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv13/Conv2D"
  input: "sub_stages/sub_conv13/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv13/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv13/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.025515519082546234
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv14/kernel"
  input: "sub_stages/sub_conv14/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv14/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_conv14/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_conv14/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_conv14/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_conv14/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_conv14/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_conv14/bias"
  input: "sub_stages/sub_conv14/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv14/bias/read"
  op: "Identity"
  input: "sub_stages/sub_conv14/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_conv14/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv13/Relu"
  input: "sub_stages/sub_conv14/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_conv14/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_conv14/Conv2D"
  input: "sub_stages/sub_conv14/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_conv14/Relu"
  op: "Relu"
  input: "sub_stages/sub_conv14/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.03227486088871956
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.03227486088871956
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/max"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/RandomUniform"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform"
  op: "Add"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/mul"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/Assign"
  op: "Assign"
  input: "sub_stages/sub_stage_img_feature/kernel"
  input: "sub_stages/sub_stage_img_feature/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/kernel/read"
  op: "Identity"
  input: "sub_stages/sub_stage_img_feature/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/bias/Initializer/zeros"
  op: "Fill"
  input: "sub_stages/sub_stage_img_feature/bias/Initializer/zeros/shape_as_tensor"
  input: "sub_stages/sub_stage_img_feature/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/bias/Assign"
  op: "Assign"
  input: "sub_stages/sub_stage_img_feature/bias"
  input: "sub_stages/sub_stage_img_feature/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/bias/read"
  op: "Identity"
  input: "sub_stages/sub_stage_img_feature/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_conv14/Relu"
  input: "sub_stages/sub_stage_img_feature/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/BiasAdd"
  op: "BiasAdd"
  input: "sub_stages/sub_stage_img_feature/Conv2D"
  input: "sub_stages/sub_stage_img_feature/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "sub_stages/sub_stage_img_feature/Relu"
  op: "Relu"
  input: "sub_stages/sub_stage_img_feature/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.09682458639144897
      }
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.09682458639144897
      }
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_1/conv1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_1/conv1/kernel/Initializer/random_uniform/max"
  input: "stage_1/conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_1/conv1/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_1/conv1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_1/conv1/kernel/Initializer/random_uniform/mul"
  input: "stage_1/conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_1/conv1/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_1/conv1/kernel/Assign"
  op: "Assign"
  input: "stage_1/conv1/kernel"
  input: "stage_1/conv1/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_1/conv1/kernel/read"
  op: "Identity"
  input: "stage_1/conv1/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_1/conv1/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "stage_1/conv1/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_1/conv1/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_1/conv1/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_1/conv1/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_1/conv1/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_1/conv1/bias/Assign"
  op: "Assign"
  input: "stage_1/conv1/bias"
  input: "stage_1/conv1/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_1/conv1/bias/read"
  op: "Identity"
  input: "stage_1/conv1/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
}
node {
  name: "stage_1/conv1/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_1/conv1/Conv2D"
  op: "Conv2D"
  input: "sub_stages/sub_stage_img_feature/Relu"
  input: "stage_1/conv1/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_1/conv1/BiasAdd"
  op: "BiasAdd"
  input: "stage_1/conv1/Conv2D"
  input: "stage_1/conv1/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_1/conv1/Relu"
  op: "Relu"
  input: "stage_1/conv1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1073140949010849
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1073140949010849
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/max"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/mul"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 512
        }
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/Assign"
  op: "Assign"
  input: "stage_1/stage_heatmap/kernel"
  input: "stage_1/stage_heatmap/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_1/stage_heatmap/kernel/read"
  op: "Identity"
  input: "stage_1/stage_heatmap/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 9
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_1/stage_heatmap/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_1/stage_heatmap/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_1/stage_heatmap/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_1/stage_heatmap/bias/Assign"
  op: "Assign"
  input: "stage_1/stage_heatmap/bias"
  input: "stage_1/stage_heatmap/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_1/stage_heatmap/bias/read"
  op: "Identity"
  input: "stage_1/stage_heatmap/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_1/stage_heatmap/Conv2D"
  op: "Conv2D"
  input: "stage_1/conv1/Relu"
  input: "stage_1/stage_heatmap/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_1/stage_heatmap/BiasAdd"
  op: "BiasAdd"
  input: "stage_1/stage_heatmap/Conv2D"
  input: "stage_1/stage_heatmap/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_2/concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "stage_2/concat"
  op: "ConcatV2"
  input: "stage_1/stage_heatmap/BiasAdd"
  input: "sub_stages/sub_stage_img_feature/Relu"
  input: "stage_2/concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.021495850756764412
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.021495850756764412
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform/max"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform/mul"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 137
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/Assign"
  op: "Assign"
  input: "stage_2/mid_conv1/kernel"
  input: "stage_2/mid_conv1/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv1/kernel/read"
  op: "Identity"
  input: "stage_2/mid_conv1/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_2/mid_conv1/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_2/mid_conv1/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv1/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv1/bias/Assign"
  op: "Assign"
  input: "stage_2/mid_conv1/bias"
  input: "stage_2/mid_conv1/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv1/bias/read"
  op: "Identity"
  input: "stage_2/mid_conv1/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv1/Conv2D"
  op: "Conv2D"
  input: "stage_2/concat"
  input: "stage_2/mid_conv1/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv1/BiasAdd"
  op: "BiasAdd"
  input: "stage_2/mid_conv1/Conv2D"
  input: "stage_2/mid_conv1/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_2/mid_conv1/Relu"
  op: "Relu"
  input: "stage_2/mid_conv1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform/max"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform/mul"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/Assign"
  op: "Assign"
  input: "stage_2/mid_conv2/kernel"
  input: "stage_2/mid_conv2/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv2/kernel/read"
  op: "Identity"
  input: "stage_2/mid_conv2/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_2/mid_conv2/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_2/mid_conv2/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv2/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv2/bias/Assign"
  op: "Assign"
  input: "stage_2/mid_conv2/bias"
  input: "stage_2/mid_conv2/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv2/bias/read"
  op: "Identity"
  input: "stage_2/mid_conv2/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv2/Conv2D"
  op: "Conv2D"
  input: "stage_2/mid_conv1/Relu"
  input: "stage_2/mid_conv2/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv2/BiasAdd"
  op: "BiasAdd"
  input: "stage_2/mid_conv2/Conv2D"
  input: "stage_2/mid_conv2/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_2/mid_conv2/Relu"
  op: "Relu"
  input: "stage_2/mid_conv2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform/max"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform/mul"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/Assign"
  op: "Assign"
  input: "stage_2/mid_conv3/kernel"
  input: "stage_2/mid_conv3/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv3/kernel/read"
  op: "Identity"
  input: "stage_2/mid_conv3/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_2/mid_conv3/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_2/mid_conv3/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv3/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv3/bias/Assign"
  op: "Assign"
  input: "stage_2/mid_conv3/bias"
  input: "stage_2/mid_conv3/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv3/bias/read"
  op: "Identity"
  input: "stage_2/mid_conv3/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv3/Conv2D"
  op: "Conv2D"
  input: "stage_2/mid_conv2/Relu"
  input: "stage_2/mid_conv3/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv3/BiasAdd"
  op: "BiasAdd"
  input: "stage_2/mid_conv3/Conv2D"
  input: "stage_2/mid_conv3/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_2/mid_conv3/Relu"
  op: "Relu"
  input: "stage_2/mid_conv3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform/max"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform/mul"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/Assign"
  op: "Assign"
  input: "stage_2/mid_conv4/kernel"
  input: "stage_2/mid_conv4/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv4/kernel/read"
  op: "Identity"
  input: "stage_2/mid_conv4/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_2/mid_conv4/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_2/mid_conv4/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv4/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv4/bias/Assign"
  op: "Assign"
  input: "stage_2/mid_conv4/bias"
  input: "stage_2/mid_conv4/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv4/bias/read"
  op: "Identity"
  input: "stage_2/mid_conv4/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv4/Conv2D"
  op: "Conv2D"
  input: "stage_2/mid_conv3/Relu"
  input: "stage_2/mid_conv4/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv4/BiasAdd"
  op: "BiasAdd"
  input: "stage_2/mid_conv4/Conv2D"
  input: "stage_2/mid_conv4/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_2/mid_conv4/Relu"
  op: "Relu"
  input: "stage_2/mid_conv4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform/max"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform/mul"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/Assign"
  op: "Assign"
  input: "stage_2/mid_conv5/kernel"
  input: "stage_2/mid_conv5/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv5/kernel/read"
  op: "Identity"
  input: "stage_2/mid_conv5/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_2/mid_conv5/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_2/mid_conv5/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv5/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv5/bias/Assign"
  op: "Assign"
  input: "stage_2/mid_conv5/bias"
  input: "stage_2/mid_conv5/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv5/bias/read"
  op: "Identity"
  input: "stage_2/mid_conv5/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv5/Conv2D"
  op: "Conv2D"
  input: "stage_2/mid_conv4/Relu"
  input: "stage_2/mid_conv5/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv5/BiasAdd"
  op: "BiasAdd"
  input: "stage_2/mid_conv5/Conv2D"
  input: "stage_2/mid_conv5/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_2/mid_conv5/Relu"
  op: "Relu"
  input: "stage_2/mid_conv5/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1530931144952774
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1530931144952774
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform/max"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform/mul"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/Assign"
  op: "Assign"
  input: "stage_2/mid_conv6/kernel"
  input: "stage_2/mid_conv6/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv6/kernel/read"
  op: "Identity"
  input: "stage_2/mid_conv6/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_2/mid_conv6/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_2/mid_conv6/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv6/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv6/bias/Assign"
  op: "Assign"
  input: "stage_2/mid_conv6/bias"
  input: "stage_2/mid_conv6/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv6/bias/read"
  op: "Identity"
  input: "stage_2/mid_conv6/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv6/Conv2D"
  op: "Conv2D"
  input: "stage_2/mid_conv5/Relu"
  input: "stage_2/mid_conv6/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv6/BiasAdd"
  op: "BiasAdd"
  input: "stage_2/mid_conv6/Conv2D"
  input: "stage_2/mid_conv6/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_2/mid_conv6/Relu"
  op: "Relu"
  input: "stage_2/mid_conv6/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.2092740386724472
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.2092740386724472
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform/max"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform/mul"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/Assign"
  op: "Assign"
  input: "stage_2/mid_conv7/kernel"
  input: "stage_2/mid_conv7/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv7/kernel/read"
  op: "Identity"
  input: "stage_2/mid_conv7/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 9
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_2/mid_conv7/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_2/mid_conv7/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_2/mid_conv7/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_2/mid_conv7/bias/Assign"
  op: "Assign"
  input: "stage_2/mid_conv7/bias"
  input: "stage_2/mid_conv7/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv7/bias/read"
  op: "Identity"
  input: "stage_2/mid_conv7/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_2/mid_conv7/Conv2D"
  op: "Conv2D"
  input: "stage_2/mid_conv6/Relu"
  input: "stage_2/mid_conv7/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_2/mid_conv7/BiasAdd"
  op: "BiasAdd"
  input: "stage_2/mid_conv7/Conv2D"
  input: "stage_2/mid_conv7/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_3/concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "stage_3/concat"
  op: "ConcatV2"
  input: "stage_2/mid_conv7/BiasAdd"
  input: "sub_stages/sub_stage_img_feature/Relu"
  input: "stage_3/concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.021495850756764412
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.021495850756764412
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform/max"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform/mul"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 137
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/Assign"
  op: "Assign"
  input: "stage_3/mid_conv1/kernel"
  input: "stage_3/mid_conv1/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv1/kernel/read"
  op: "Identity"
  input: "stage_3/mid_conv1/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_3/mid_conv1/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_3/mid_conv1/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv1/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv1/bias/Assign"
  op: "Assign"
  input: "stage_3/mid_conv1/bias"
  input: "stage_3/mid_conv1/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv1/bias/read"
  op: "Identity"
  input: "stage_3/mid_conv1/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv1/Conv2D"
  op: "Conv2D"
  input: "stage_3/concat"
  input: "stage_3/mid_conv1/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv1/BiasAdd"
  op: "BiasAdd"
  input: "stage_3/mid_conv1/Conv2D"
  input: "stage_3/mid_conv1/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_3/mid_conv1/Relu"
  op: "Relu"
  input: "stage_3/mid_conv1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform/max"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform/mul"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/Assign"
  op: "Assign"
  input: "stage_3/mid_conv2/kernel"
  input: "stage_3/mid_conv2/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv2/kernel/read"
  op: "Identity"
  input: "stage_3/mid_conv2/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_3/mid_conv2/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_3/mid_conv2/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv2/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv2/bias/Assign"
  op: "Assign"
  input: "stage_3/mid_conv2/bias"
  input: "stage_3/mid_conv2/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv2/bias/read"
  op: "Identity"
  input: "stage_3/mid_conv2/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv2/Conv2D"
  op: "Conv2D"
  input: "stage_3/mid_conv1/Relu"
  input: "stage_3/mid_conv2/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv2/BiasAdd"
  op: "BiasAdd"
  input: "stage_3/mid_conv2/Conv2D"
  input: "stage_3/mid_conv2/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_3/mid_conv2/Relu"
  op: "Relu"
  input: "stage_3/mid_conv2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform/max"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform/mul"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/Assign"
  op: "Assign"
  input: "stage_3/mid_conv3/kernel"
  input: "stage_3/mid_conv3/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv3/kernel/read"
  op: "Identity"
  input: "stage_3/mid_conv3/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_3/mid_conv3/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_3/mid_conv3/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv3/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv3/bias/Assign"
  op: "Assign"
  input: "stage_3/mid_conv3/bias"
  input: "stage_3/mid_conv3/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv3/bias/read"
  op: "Identity"
  input: "stage_3/mid_conv3/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv3/Conv2D"
  op: "Conv2D"
  input: "stage_3/mid_conv2/Relu"
  input: "stage_3/mid_conv3/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv3/BiasAdd"
  op: "BiasAdd"
  input: "stage_3/mid_conv3/Conv2D"
  input: "stage_3/mid_conv3/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_3/mid_conv3/Relu"
  op: "Relu"
  input: "stage_3/mid_conv3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform/max"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform/mul"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/Assign"
  op: "Assign"
  input: "stage_3/mid_conv4/kernel"
  input: "stage_3/mid_conv4/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv4/kernel/read"
  op: "Identity"
  input: "stage_3/mid_conv4/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_3/mid_conv4/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_3/mid_conv4/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv4/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv4/bias/Assign"
  op: "Assign"
  input: "stage_3/mid_conv4/bias"
  input: "stage_3/mid_conv4/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv4/bias/read"
  op: "Identity"
  input: "stage_3/mid_conv4/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv4/Conv2D"
  op: "Conv2D"
  input: "stage_3/mid_conv3/Relu"
  input: "stage_3/mid_conv4/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv4/BiasAdd"
  op: "BiasAdd"
  input: "stage_3/mid_conv4/Conv2D"
  input: "stage_3/mid_conv4/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_3/mid_conv4/Relu"
  op: "Relu"
  input: "stage_3/mid_conv4/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.02187044359743595
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform/max"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform/mul"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/Assign"
  op: "Assign"
  input: "stage_3/mid_conv5/kernel"
  input: "stage_3/mid_conv5/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv5/kernel/read"
  op: "Identity"
  input: "stage_3/mid_conv5/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_3/mid_conv5/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_3/mid_conv5/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv5/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv5/bias/Assign"
  op: "Assign"
  input: "stage_3/mid_conv5/bias"
  input: "stage_3/mid_conv5/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv5/bias/read"
  op: "Identity"
  input: "stage_3/mid_conv5/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv5/bias"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv5/Conv2D"
  op: "Conv2D"
  input: "stage_3/mid_conv4/Relu"
  input: "stage_3/mid_conv5/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv5/BiasAdd"
  op: "BiasAdd"
  input: "stage_3/mid_conv5/Conv2D"
  input: "stage_3/mid_conv5/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_3/mid_conv5/Relu"
  op: "Relu"
  input: "stage_3/mid_conv5/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1530931144952774
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1530931144952774
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform/max"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform/mul"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/Assign"
  op: "Assign"
  input: "stage_3/mid_conv6/kernel"
  input: "stage_3/mid_conv6/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv6/kernel/read"
  op: "Identity"
  input: "stage_3/mid_conv6/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_3/mid_conv6/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_3/mid_conv6/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv6/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv6/bias/Assign"
  op: "Assign"
  input: "stage_3/mid_conv6/bias"
  input: "stage_3/mid_conv6/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv6/bias/read"
  op: "Identity"
  input: "stage_3/mid_conv6/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv6/bias"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv6/Conv2D"
  op: "Conv2D"
  input: "stage_3/mid_conv5/Relu"
  input: "stage_3/mid_conv6/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv6/BiasAdd"
  op: "BiasAdd"
  input: "stage_3/mid_conv6/Conv2D"
  input: "stage_3/mid_conv6/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "stage_3/mid_conv6/Relu"
  op: "Relu"
  input: "stage_3/mid_conv6/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.2092740386724472
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.2092740386724472
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform/max"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform/RandomUniform"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Initializer/random_uniform"
  op: "Add"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform/mul"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/Assign"
  op: "Assign"
  input: "stage_3/mid_conv7/kernel"
  input: "stage_3/mid_conv7/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv7/kernel/read"
  op: "Identity"
  input: "stage_3/mid_conv7/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/bias/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 9
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/bias/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/bias/Initializer/zeros"
  op: "Fill"
  input: "stage_3/mid_conv7/bias/Initializer/zeros/shape_as_tensor"
  input: "stage_3/mid_conv7/bias/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage_3/mid_conv7/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "stage_3/mid_conv7/bias/Assign"
  op: "Assign"
  input: "stage_3/mid_conv7/bias"
  input: "stage_3/mid_conv7/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv7/bias/read"
  op: "Identity"
  input: "stage_3/mid_conv7/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv7/bias"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "stage_3/mid_conv7/Conv2D"
  op: "Conv2D"
  input: "stage_3/mid_conv6/Relu"
  input: "stage_3/mid_conv7/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "stage_3/mid_conv7/BiasAdd"
  op: "BiasAdd"
  input: "stage_3/mid_conv7/Conv2D"
  input: "stage_3/mid_conv7/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "Shape"
  op: "Shape"
  input: "input_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "strided_slice"
  op: "StridedSlice"
  input: "Shape"
  input: "strided_slice/stack"
  input: "strided_slice/stack_1"
  input: "strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "Cast"
  op: "Cast"
  input: "strided_slice"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "stage1_loss/sub"
  op: "Sub"
  input: "stage_1/stage_heatmap/BiasAdd"
  input: "gt_hmap_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage1_loss/l2_loss"
  op: "L2Loss"
  input: "stage1_loss/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage1_loss/truediv"
  op: "RealDiv"
  input: "stage1_loss/l2_loss"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage1_loss_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "stage1_loss_1"
      }
    }
  }
}
node {
  name: "stage1_loss_1"
  op: "ScalarSummary"
  input: "stage1_loss_1/tags"
  input: "stage1_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage2_loss/sub"
  op: "Sub"
  input: "stage_2/mid_conv7/BiasAdd"
  input: "gt_hmap_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage2_loss/l2_loss"
  op: "L2Loss"
  input: "stage2_loss/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage2_loss/truediv"
  op: "RealDiv"
  input: "stage2_loss/l2_loss"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage2_loss_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "stage2_loss_1"
      }
    }
  }
}
node {
  name: "stage2_loss_1"
  op: "ScalarSummary"
  input: "stage2_loss_1/tags"
  input: "stage2_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage3_loss/sub"
  op: "Sub"
  input: "stage_3/mid_conv7/BiasAdd"
  input: "gt_hmap_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage3_loss/l2_loss"
  op: "L2Loss"
  input: "stage3_loss/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage3_loss/truediv"
  op: "RealDiv"
  input: "stage3_loss/l2_loss"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "stage3_loss_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "stage3_loss_1"
      }
    }
  }
}
node {
  name: "stage3_loss_1"
  op: "ScalarSummary"
  input: "stage3_loss_1/tags"
  input: "stage3_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss/add/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "total_loss/add"
  op: "Add"
  input: "total_loss/add/x"
  input: "stage1_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss/add_1"
  op: "Add"
  input: "total_loss/add"
  input: "stage2_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss/add_2"
  op: "Add"
  input: "total_loss/add_1"
  input: "stage3_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss/total_loss_train/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "total_loss/total_loss_train"
      }
    }
  }
}
node {
  name: "total_loss/total_loss_train"
  op: "ScalarSummary"
  input: "total_loss/total_loss_train/tags"
  input: "total_loss/add_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss_eval/add/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "total_loss_eval/add"
  op: "Add"
  input: "total_loss_eval/add/x"
  input: "stage1_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss_eval/add_1"
  op: "Add"
  input: "total_loss_eval/add"
  input: "stage2_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss_eval/add_2"
  op: "Add"
  input: "total_loss_eval/add_1"
  input: "stage3_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "total_loss_eval/total_loss_eval/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "total_loss_eval/total_loss_eval"
      }
    }
  }
}
node {
  name: "total_loss_eval/total_loss_eval"
  op: "ScalarSummary"
  input: "total_loss_eval/total_loss_eval/tags"
  input: "total_loss/add_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/global_step/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/global_step"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/global_step/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/global_step"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT64
        tensor_shape {
        }
        int64_val: 0
      }
    }
  }
}
node {
  name: "train/global_step/Initializer/zeros"
  op: "Fill"
  input: "train/global_step/Initializer/zeros/shape_as_tensor"
  input: "train/global_step/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/global_step"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/global_step"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/global_step"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/global_step/Assign"
  op: "Assign"
  input: "train/global_step"
  input: "train/global_step/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/global_step"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/global_step/read"
  op: "Identity"
  input: "train/global_step"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/global_step"
      }
    }
  }
}
node {
  name: "train/ExponentialDecay/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "train/ExponentialDecay/Cast"
  op: "Cast"
  input: "train/global_step/read"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "train/ExponentialDecay/Cast_1/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 5000
      }
    }
  }
}
node {
  name: "train/ExponentialDecay/Cast_1"
  op: "Cast"
  input: "train/ExponentialDecay/Cast_1/x"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/ExponentialDecay/Cast_2/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "train/ExponentialDecay/truediv"
  op: "RealDiv"
  input: "train/ExponentialDecay/Cast"
  input: "train/ExponentialDecay/Cast_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/ExponentialDecay/Pow"
  op: "Pow"
  input: "train/ExponentialDecay/Cast_2/x"
  input: "train/ExponentialDecay/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/ExponentialDecay"
  op: "Mul"
  input: "train/ExponentialDecay/learning_rate"
  input: "train/ExponentialDecay/Pow"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/global_learning_rate/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "train/global_learning_rate"
      }
    }
  }
}
node {
  name: "train/global_learning_rate"
  op: "ScalarSummary"
  input: "train/global_learning_rate/tags"
  input: "train/ExponentialDecay"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/grad_ys_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/Fill"
  op: "Fill"
  input: "train/OptimizeLoss/gradients/Shape"
  input: "train/OptimizeLoss/gradients/grad_ys_0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/Fill"
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/Fill"
  input: "^train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/Fill"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/Fill"
  input: "^train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/Fill"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/control_dependency"
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/control_dependency"
  input: "^train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/Fill"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/control_dependency"
  input: "^train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/Fill"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Shape"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/RealDiv"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/control_dependency_1"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/RealDiv"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Sum"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Neg"
  op: "Neg"
  input: "stage3_loss/l2_loss"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/RealDiv_1"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Neg"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/RealDiv_2"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/RealDiv_1"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/mul"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/total_loss/add_2_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Sum_1"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/mul"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape_1"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Sum_1"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage3_loss/truediv_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape_1"
  input: "^train/OptimizeLoss/gradients/stage3_loss/truediv_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage3_loss/truediv_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/control_dependency"
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/control_dependency"
  input: "^train/OptimizeLoss/gradients/total_loss/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/Fill"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/total_loss/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/control_dependency"
  input: "^train/OptimizeLoss/gradients/total_loss/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/Fill"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Shape"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/RealDiv"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/control_dependency_1"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/RealDiv"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Sum"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Neg"
  op: "Neg"
  input: "stage2_loss/l2_loss"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/RealDiv_1"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Neg"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/RealDiv_2"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/RealDiv_1"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/mul"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/total_loss/add_1_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Sum_1"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/mul"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape_1"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Sum_1"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage2_loss/truediv_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape_1"
  input: "^train/OptimizeLoss/gradients/stage2_loss/truediv_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage2_loss/truediv_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/l2_loss_grad/mul"
  op: "Mul"
  input: "stage3_loss/sub"
  input: "train/OptimizeLoss/gradients/stage3_loss/truediv_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Shape"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/RealDiv"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/total_loss/add_grad/tuple/control_dependency_1"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/RealDiv"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Sum"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Neg"
  op: "Neg"
  input: "stage1_loss/l2_loss"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/RealDiv_1"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Neg"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/RealDiv_2"
  op: "RealDiv"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/RealDiv_1"
  input: "Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/mul"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/total_loss/add_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Sum_1"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/mul"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape_1"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Sum_1"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage1_loss/truediv_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape_1"
  input: "^train/OptimizeLoss/gradients/stage1_loss/truediv_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage1_loss/truediv_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/l2_loss_grad/mul"
  op: "Mul"
  input: "stage2_loss/sub"
  input: "train/OptimizeLoss/gradients/stage2_loss/truediv_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Shape"
  op: "Shape"
  input: "stage_3/mid_conv7/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Shape_1"
  op: "Shape"
  input: "gt_hmap_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Shape"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage3_loss/l2_loss_grad/mul"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Sum"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Sum_1"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage3_loss/l2_loss_grad/mul"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Neg"
  op: "Neg"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape_1"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Neg"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape_1"
  input: "^train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/l2_loss_grad/mul"
  op: "Mul"
  input: "stage1_loss/sub"
  input: "train/OptimizeLoss/gradients/stage1_loss/truediv_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Shape"
  op: "Shape"
  input: "stage_2/mid_conv7/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Shape_1"
  op: "Shape"
  input: "gt_hmap_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Shape"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage2_loss/l2_loss_grad/mul"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Sum"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Sum_1"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage2_loss/l2_loss_grad/mul"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Neg"
  op: "Neg"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape_1"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Neg"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage2_loss/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape_1"
  input: "^train/OptimizeLoss/gradients/stage2_loss/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/control_dependency"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage3_loss/sub_grad/tuple/control_dependency"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage3_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Shape"
  op: "Shape"
  input: "stage_1/stage_heatmap/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Shape_1"
  op: "Shape"
  input: "gt_hmap_placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Shape"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage1_loss/l2_loss_grad/mul"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Sum"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Sum_1"
  op: "Sum"
  input: "train/OptimizeLoss/gradients/stage1_loss/l2_loss_grad/mul"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Neg"
  op: "Neg"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape_1"
  op: "Reshape"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Neg"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape"
  input: "^train/OptimizeLoss/gradients/stage1_loss/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape_1"
  input: "^train/OptimizeLoss/gradients/stage1_loss/sub_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_3/mid_conv6/Relu"
  input: "stage_3/mid_conv7/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/ShapeN"
  input: "stage_3/mid_conv7/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_3/mid_conv6/Relu"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/control_dependency"
  input: "stage_3/mid_conv6/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_3/mid_conv5/Relu"
  input: "stage_3/mid_conv6/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/ShapeN"
  input: "stage_3/mid_conv6/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_3/mid_conv5/Relu"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/control_dependency"
  input: "stage_3/mid_conv5/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_3/mid_conv4/Relu"
  input: "stage_3/mid_conv5/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/ShapeN"
  input: "stage_3/mid_conv5/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_3/mid_conv4/Relu"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/control_dependency"
  input: "stage_3/mid_conv4/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_3/mid_conv3/Relu"
  input: "stage_3/mid_conv4/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/ShapeN"
  input: "stage_3/mid_conv4/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_3/mid_conv3/Relu"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/control_dependency"
  input: "stage_3/mid_conv3/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_3/mid_conv2/Relu"
  input: "stage_3/mid_conv3/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/ShapeN"
  input: "stage_3/mid_conv3/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_3/mid_conv2/Relu"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/control_dependency"
  input: "stage_3/mid_conv2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_3/mid_conv1/Relu"
  input: "stage_3/mid_conv2/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/ShapeN"
  input: "stage_3/mid_conv2/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_3/mid_conv1/Relu"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/control_dependency"
  input: "stage_3/mid_conv1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_3/concat"
  input: "stage_3/mid_conv1/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/ShapeN"
  input: "stage_3/mid_conv1/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_3/concat"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/mod"
  op: "FloorMod"
  input: "stage_3/concat/axis"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/Rank"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/Shape"
  op: "Shape"
  input: "stage_2/mid_conv7/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/mid_conv7/BiasAdd"
  input: "sub_stages/sub_stage_img_feature/Relu"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/ConcatOffset"
  op: "ConcatOffset"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/mod"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/ShapeN"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/ShapeN:1"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/Slice"
  op: "Slice"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/control_dependency"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/ConcatOffset"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/ShapeN"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/Slice_1"
  op: "Slice"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/control_dependency"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/ConcatOffset:1"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/ShapeN:1"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_3/concat_grad/Slice"
  input: "^train/OptimizeLoss/gradients/stage_3/concat_grad/Slice_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/Slice"
  input: "^train/OptimizeLoss/gradients/stage_3/concat_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/concat_grad/Slice"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_3/concat_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/Slice_1"
  input: "^train/OptimizeLoss/gradients/stage_3/concat_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/concat_grad/Slice_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/AddN"
  op: "AddN"
  input: "train/OptimizeLoss/gradients/stage2_loss/sub_grad/tuple/control_dependency"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/tuple/control_dependency"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/AddN"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/AddN"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/AddN"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage2_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/mid_conv6/Relu"
  input: "stage_2/mid_conv7/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/ShapeN"
  input: "stage_2/mid_conv7/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_2/mid_conv6/Relu"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/control_dependency"
  input: "stage_2/mid_conv6/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/mid_conv5/Relu"
  input: "stage_2/mid_conv6/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/ShapeN"
  input: "stage_2/mid_conv6/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_2/mid_conv5/Relu"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/control_dependency"
  input: "stage_2/mid_conv5/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/mid_conv4/Relu"
  input: "stage_2/mid_conv5/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/ShapeN"
  input: "stage_2/mid_conv5/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_2/mid_conv4/Relu"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/control_dependency"
  input: "stage_2/mid_conv4/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/mid_conv3/Relu"
  input: "stage_2/mid_conv4/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/ShapeN"
  input: "stage_2/mid_conv4/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_2/mid_conv3/Relu"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/control_dependency"
  input: "stage_2/mid_conv3/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/mid_conv2/Relu"
  input: "stage_2/mid_conv3/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/ShapeN"
  input: "stage_2/mid_conv3/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_2/mid_conv2/Relu"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/control_dependency"
  input: "stage_2/mid_conv2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/mid_conv1/Relu"
  input: "stage_2/mid_conv2/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/ShapeN"
  input: "stage_2/mid_conv2/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_2/mid_conv1/Relu"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/control_dependency"
  input: "stage_2/mid_conv1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_2/concat"
  input: "stage_2/mid_conv1/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/ShapeN"
  input: "stage_2/mid_conv1/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_2/concat"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/mod"
  op: "FloorMod"
  input: "stage_2/concat/axis"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/Rank"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/Shape"
  op: "Shape"
  input: "stage_1/stage_heatmap/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/ShapeN"
  op: "ShapeN"
  input: "stage_1/stage_heatmap/BiasAdd"
  input: "sub_stages/sub_stage_img_feature/Relu"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/ConcatOffset"
  op: "ConcatOffset"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/mod"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/ShapeN"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/ShapeN:1"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/Slice"
  op: "Slice"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/control_dependency"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/ConcatOffset"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/ShapeN"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/Slice_1"
  op: "Slice"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/control_dependency"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/ConcatOffset:1"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/ShapeN:1"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_2/concat_grad/Slice"
  input: "^train/OptimizeLoss/gradients/stage_2/concat_grad/Slice_1"
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/Slice"
  input: "^train/OptimizeLoss/gradients/stage_2/concat_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/concat_grad/Slice"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_2/concat_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/Slice_1"
  input: "^train/OptimizeLoss/gradients/stage_2/concat_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/concat_grad/Slice_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/AddN_1"
  op: "AddN"
  input: "train/OptimizeLoss/gradients/stage1_loss/sub_grad/tuple/control_dependency"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/tuple/control_dependency"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/AddN_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/AddN_1"
  input: "^train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/AddN_1"
  input: "^train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage1_loss/sub_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "stage_1/conv1/Relu"
  input: "stage_1/stage_heatmap/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/ShapeN"
  input: "stage_1/stage_heatmap/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "stage_1/conv1/Relu"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/control_dependency"
  input: "stage_1/conv1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_stage_img_feature/Relu"
  input: "stage_1/conv1/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/ShapeN"
  input: "stage_1/conv1/kernel/read"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_stage_img_feature/Relu"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/AddN_2"
  op: "AddN"
  input: "train/OptimizeLoss/gradients/stage_3/concat_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/gradients/stage_2/concat_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/control_dependency"
  attr {
    key: "N"
    value {
      i: 3
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/concat_grad/Slice_1"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/AddN_2"
  input: "sub_stages/sub_stage_img_feature/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv14/Relu"
  input: "sub_stages/sub_stage_img_feature/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_stage_img_feature/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv14/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv14/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv13/Relu"
  input: "sub_stages/sub_conv14/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv14/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv13/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv13/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv12/Relu"
  input: "sub_stages/sub_conv13/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv13/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv12/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv12/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv11/Relu"
  input: "sub_stages/sub_conv12/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv12/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv11/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv11/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv10/Relu"
  input: "sub_stages/sub_conv11/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv11/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv10/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv10/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv9/Relu"
  input: "sub_stages/sub_conv10/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv10/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv9/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv9/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_pool3/MaxPool"
  input: "sub_stages/sub_conv9/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv9/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_pool3/MaxPool"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_pool3/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "sub_stages/sub_conv8/Relu"
  input: "sub_stages/sub_pool3/MaxPool"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_pool3/MaxPool_grad/MaxPoolGrad"
  input: "sub_stages/sub_conv8/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv7/Relu"
  input: "sub_stages/sub_conv8/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv8/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv7/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv7/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv6/Relu"
  input: "sub_stages/sub_conv7/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv7/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv6/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv6/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv5/Relu"
  input: "sub_stages/sub_conv6/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv6/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv5/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv5/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_pool2/MaxPool"
  input: "sub_stages/sub_conv5/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv5/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_pool2/MaxPool"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_pool2/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "sub_stages/sub_conv4/Relu"
  input: "sub_stages/sub_pool2/MaxPool"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_pool2/MaxPool_grad/MaxPoolGrad"
  input: "sub_stages/sub_conv4/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv3/Relu"
  input: "sub_stages/sub_conv4/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv4/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv3/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv3/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_pool1/MaxPool"
  input: "sub_stages/sub_conv3/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv3/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_pool1/MaxPool"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_pool1/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "sub_stages/sub_conv2/Relu"
  input: "sub_stages/sub_pool1/MaxPool"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_pool1/MaxPool_grad/MaxPoolGrad"
  input: "sub_stages/sub_conv2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "sub_stages/sub_conv1/Relu"
  input: "sub_stages/sub_conv2/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv2/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "sub_stages/sub_conv1/Relu"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/control_dependency"
  input: "sub_stages/sub_conv1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Relu_grad/ReluGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "input_placeholder"
  input: "sub_stages/sub_conv1/kernel/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\003\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/ShapeN"
  input: "sub_stages/sub_conv1/kernel/read"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "input_placeholder"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Const"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropInput"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_1"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_2"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_3"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_4"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_5"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_6"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_7"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_8"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_9"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_10"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_11"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_12"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_13"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_14"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_15"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_16"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_17"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_18"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_19"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_20"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_21"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_22"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_23"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_24"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_25"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_26"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_27"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_28"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_29"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_30"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_31"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_32"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_33"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_34"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_35"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_36"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_37"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_38"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_39"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_40"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_41"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_42"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_43"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_44"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_45"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_46"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_47"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_48"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_49"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_50"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_51"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_52"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_53"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_54"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_55"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_56"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_57"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_58"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_59"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_60"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/L2Loss_61"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/stack"
  op: "Pack"
  input: "train/OptimizeLoss/global_norm/L2Loss"
  input: "train/OptimizeLoss/global_norm/L2Loss_1"
  input: "train/OptimizeLoss/global_norm/L2Loss_2"
  input: "train/OptimizeLoss/global_norm/L2Loss_3"
  input: "train/OptimizeLoss/global_norm/L2Loss_4"
  input: "train/OptimizeLoss/global_norm/L2Loss_5"
  input: "train/OptimizeLoss/global_norm/L2Loss_6"
  input: "train/OptimizeLoss/global_norm/L2Loss_7"
  input: "train/OptimizeLoss/global_norm/L2Loss_8"
  input: "train/OptimizeLoss/global_norm/L2Loss_9"
  input: "train/OptimizeLoss/global_norm/L2Loss_10"
  input: "train/OptimizeLoss/global_norm/L2Loss_11"
  input: "train/OptimizeLoss/global_norm/L2Loss_12"
  input: "train/OptimizeLoss/global_norm/L2Loss_13"
  input: "train/OptimizeLoss/global_norm/L2Loss_14"
  input: "train/OptimizeLoss/global_norm/L2Loss_15"
  input: "train/OptimizeLoss/global_norm/L2Loss_16"
  input: "train/OptimizeLoss/global_norm/L2Loss_17"
  input: "train/OptimizeLoss/global_norm/L2Loss_18"
  input: "train/OptimizeLoss/global_norm/L2Loss_19"
  input: "train/OptimizeLoss/global_norm/L2Loss_20"
  input: "train/OptimizeLoss/global_norm/L2Loss_21"
  input: "train/OptimizeLoss/global_norm/L2Loss_22"
  input: "train/OptimizeLoss/global_norm/L2Loss_23"
  input: "train/OptimizeLoss/global_norm/L2Loss_24"
  input: "train/OptimizeLoss/global_norm/L2Loss_25"
  input: "train/OptimizeLoss/global_norm/L2Loss_26"
  input: "train/OptimizeLoss/global_norm/L2Loss_27"
  input: "train/OptimizeLoss/global_norm/L2Loss_28"
  input: "train/OptimizeLoss/global_norm/L2Loss_29"
  input: "train/OptimizeLoss/global_norm/L2Loss_30"
  input: "train/OptimizeLoss/global_norm/L2Loss_31"
  input: "train/OptimizeLoss/global_norm/L2Loss_32"
  input: "train/OptimizeLoss/global_norm/L2Loss_33"
  input: "train/OptimizeLoss/global_norm/L2Loss_34"
  input: "train/OptimizeLoss/global_norm/L2Loss_35"
  input: "train/OptimizeLoss/global_norm/L2Loss_36"
  input: "train/OptimizeLoss/global_norm/L2Loss_37"
  input: "train/OptimizeLoss/global_norm/L2Loss_38"
  input: "train/OptimizeLoss/global_norm/L2Loss_39"
  input: "train/OptimizeLoss/global_norm/L2Loss_40"
  input: "train/OptimizeLoss/global_norm/L2Loss_41"
  input: "train/OptimizeLoss/global_norm/L2Loss_42"
  input: "train/OptimizeLoss/global_norm/L2Loss_43"
  input: "train/OptimizeLoss/global_norm/L2Loss_44"
  input: "train/OptimizeLoss/global_norm/L2Loss_45"
  input: "train/OptimizeLoss/global_norm/L2Loss_46"
  input: "train/OptimizeLoss/global_norm/L2Loss_47"
  input: "train/OptimizeLoss/global_norm/L2Loss_48"
  input: "train/OptimizeLoss/global_norm/L2Loss_49"
  input: "train/OptimizeLoss/global_norm/L2Loss_50"
  input: "train/OptimizeLoss/global_norm/L2Loss_51"
  input: "train/OptimizeLoss/global_norm/L2Loss_52"
  input: "train/OptimizeLoss/global_norm/L2Loss_53"
  input: "train/OptimizeLoss/global_norm/L2Loss_54"
  input: "train/OptimizeLoss/global_norm/L2Loss_55"
  input: "train/OptimizeLoss/global_norm/L2Loss_56"
  input: "train/OptimizeLoss/global_norm/L2Loss_57"
  input: "train/OptimizeLoss/global_norm/L2Loss_58"
  input: "train/OptimizeLoss/global_norm/L2Loss_59"
  input: "train/OptimizeLoss/global_norm/L2Loss_60"
  input: "train/OptimizeLoss/global_norm/L2Loss_61"
  attr {
    key: "N"
    value {
      i: 62
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/global_norm/stack"
  input: "train/OptimizeLoss/global_norm/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/mul"
  op: "Mul"
  input: "train/OptimizeLoss/global_norm/Sum"
  input: "train/OptimizeLoss/global_norm/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/global_norm"
  op: "Sqrt"
  input: "train/OptimizeLoss/global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/gradient_norm/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "train/OptimizeLoss/global_norm/gradient_norm"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/gradient_norm"
  op: "ScalarSummary"
  input: "train/OptimizeLoss/global_norm/gradient_norm/tags"
  input: "train/OptimizeLoss/global_norm/global_norm"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_1"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_2"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_3"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_4"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_5"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_6"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_7"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_8"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_9"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_10"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_11"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_12"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_13"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_14"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_15"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_16"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_17"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_18"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_19"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_20"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_21"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_22"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_23"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_24"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_25"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_26"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_27"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_28"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_29"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_30"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_31"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_32"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_33"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_34"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_35"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_36"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_37"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_38"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_39"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_40"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_41"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_42"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_43"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_44"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_45"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_46"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_47"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_48"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_49"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_50"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_51"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_52"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_53"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_54"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_55"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_56"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_57"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_58"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_59"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_60"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/L2Loss_61"
  op: "L2Loss"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/stack"
  op: "Pack"
  input: "train/OptimizeLoss/global_norm_1/L2Loss"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_1"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_2"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_3"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_4"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_5"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_6"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_7"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_8"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_9"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_10"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_11"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_12"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_13"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_14"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_15"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_16"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_17"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_18"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_19"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_20"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_21"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_22"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_23"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_24"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_25"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_26"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_27"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_28"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_29"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_30"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_31"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_32"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_33"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_34"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_35"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_36"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_37"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_38"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_39"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_40"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_41"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_42"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_43"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_44"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_45"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_46"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_47"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_48"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_49"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_50"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_51"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_52"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_53"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_54"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_55"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_56"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_57"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_58"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_59"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_60"
  input: "train/OptimizeLoss/global_norm_1/L2Loss_61"
  attr {
    key: "N"
    value {
      i: 62
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/global_norm_1/stack"
  input: "train/OptimizeLoss/global_norm_1/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/mul"
  op: "Mul"
  input: "train/OptimizeLoss/global_norm_1/Sum"
  input: "train/OptimizeLoss/global_norm_1/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_1/global_norm"
  op: "Sqrt"
  input: "train/OptimizeLoss/global_norm_1/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/truediv/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/truediv"
  op: "RealDiv"
  input: "train/OptimizeLoss/clip_by_global_norm/truediv/x"
  input: "train/OptimizeLoss/global_norm_1/global_norm"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/truediv_1"
  op: "RealDiv"
  input: "train/OptimizeLoss/clip_by_global_norm/Const"
  input: "train/OptimizeLoss/clip_by_global_norm/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/Minimum"
  op: "Minimum"
  input: "train/OptimizeLoss/clip_by_global_norm/truediv"
  input: "train/OptimizeLoss/clip_by_global_norm/truediv_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul"
  op: "Mul"
  input: "train/OptimizeLoss/clip_by_global_norm/mul/x"
  input: "train/OptimizeLoss/clip_by_global_norm/Minimum"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_1"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_0"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_2"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_1"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_3"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_2"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_4"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_3"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_5"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_4"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_6"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_5"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_7"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_6"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_8"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_7"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_9"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_8"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_10"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_9"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_11"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_10"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_12"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_11"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_13"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_12"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_14"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_13"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_15"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_14"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_16"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_15"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_17"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_16"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_18"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_17"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_19"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_18"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_20"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_19"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_21"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_20"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_22"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_21"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_23"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_22"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_24"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_23"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_25"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_24"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_26"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_25"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_26"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_27"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_26"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_28"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_27"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_28"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_29"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_28"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_30"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_29"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_30"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_31"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_30"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_31"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_32"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_31"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_32"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_33"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_32"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_33"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_34"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_33"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_34"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_35"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_34"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_35"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_36"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_35"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_36"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_37"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_36"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_37"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_38"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_37"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_38"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_39"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_38"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_39"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_40"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_39"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_40"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_41"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_40"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_41"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_42"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_41"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_42"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_43"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_42"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_43"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_44"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_43"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_44"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_45"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_44"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_45"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_46"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_45"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_46"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_47"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_46"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_47"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_48"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_47"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_48"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_49"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_48"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_49"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_50"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_49"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_50"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_51"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_50"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_51"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_52"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_51"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_52"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_53"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_52"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_53"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_54"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_53"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_54"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_55"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_54"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_55"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_56"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_55"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_56"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_57"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_56"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_57"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_58"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_57"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_58"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_59"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_58"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_59"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_60"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_59"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_60"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_61"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_60"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_61"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/mul_62"
  op: "Mul"
  input: "train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/tuple/control_dependency_1"
  input: "train/OptimizeLoss/clip_by_global_norm/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_61"
  op: "Identity"
  input: "train/OptimizeLoss/clip_by_global_norm/mul_62"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/loss/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "train/OptimizeLoss/loss"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/loss"
  op: "ScalarSummary"
  input: "train/OptimizeLoss/loss/tags"
  input: "total_loss/add_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_1"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_2"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_3"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_4"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_5"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_6"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_7"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_8"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_9"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_10"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_11"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_12"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_13"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_14"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_15"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv8/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_16"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_17"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv9/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_18"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_19"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv10/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_20"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_21"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv11/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_22"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_23"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv12/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_24"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_25"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv13/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_26"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_26"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_27"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_conv14/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_28"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_28"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_29"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/sub_stages/sub_stage_img_feature/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_30"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_30"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_31"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_31"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_32"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_32"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_33"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_33"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_1/stage_heatmap/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_34"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_34"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_35"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_35"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_36"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_36"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_37"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_37"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_38"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_38"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_39"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_39"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_40"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_40"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_41"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_41"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_42"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_42"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_43"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_43"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_44"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_44"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_45"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_45"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_46"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_46"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_47"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_47"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_2/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_48"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_48"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_49"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_49"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_50"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_50"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_51"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_51"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_52"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_52"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_53"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_53"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv3/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_54"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_54"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_55"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_55"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv4/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_56"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_56"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_57"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_57"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv5/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_58"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_58"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_59"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_59"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv6/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_60"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_60"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/L2Loss_61"
  op: "L2Loss"
  input: "train/OptimizeLoss/clip_by_global_norm/train/OptimizeLoss/clip_by_global_norm/_61"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/OptimizeLoss/gradients/stage_3/mid_conv7/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/stack"
  op: "Pack"
  input: "train/OptimizeLoss/global_norm_2/L2Loss"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_1"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_2"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_3"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_4"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_5"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_6"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_7"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_8"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_9"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_10"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_11"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_12"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_13"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_14"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_15"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_16"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_17"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_18"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_19"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_20"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_21"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_22"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_23"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_24"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_25"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_26"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_27"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_28"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_29"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_30"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_31"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_32"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_33"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_34"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_35"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_36"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_37"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_38"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_39"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_40"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_41"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_42"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_43"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_44"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_45"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_46"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_47"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_48"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_49"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_50"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_51"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_52"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_53"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_54"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_55"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_56"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_57"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_58"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_59"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_60"
  input: "train/OptimizeLoss/global_norm_2/L2Loss_61"
  attr {
    key: "N"
    value {
      i: 62
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/Sum"
  op: "Sum"
  input: "train/OptimizeLoss/global_norm_2/stack"
  input: "train/OptimizeLoss/global_norm_2/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/mul"
  op: "Mul"
  input: "train/OptimizeLoss/global_norm_2/Sum"
  input: "train/OptimizeLoss/global_norm_2/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm_2/global_norm"
  op: "Sqrt"
  input: "train/OptimizeLoss/global_norm_2/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/clipped_gradient_norm/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "train/OptimizeLoss/global_norm/clipped_gradient_norm"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/global_norm/clipped_gradient_norm"
  op: "ScalarSummary"
  input: "train/OptimizeLoss/global_norm/clipped_gradient_norm/tags"
  input: "train/OptimizeLoss/global_norm_2/global_norm"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\003\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\003\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv1/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv2/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv2/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv3/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv3/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv4/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv4/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv5/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv5/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv6/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv6/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv7/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv7/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\001\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 256
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 256
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv8/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv8/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\001\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 256
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv9/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv9/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv10/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv10/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv11/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv11/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv12/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv12/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv13/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv13/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_conv14/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_conv14/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\000\002\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 512
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/sub_stages/sub_stage_img_feature/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@sub_stages/sub_stage_img_feature/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\000\002\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/conv1/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 512
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 512
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/conv1/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 512
        }
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 512
        }
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 9
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 9
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_1/stage_heatmap/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_1/stage_heatmap/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 137
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 137
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv1/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv1/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv2/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv2/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv2/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv3/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv3/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv3/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv4/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv4/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv4/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv5/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv5/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv5/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv6/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv6/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv6/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\t\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 128
        }
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv7/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 9
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 9
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 9
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_2/mid_conv7/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_2/mid_conv7/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 137
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\211\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 137
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv1/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv1/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv1/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv2/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv2/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv2/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv3/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv3/bias/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv3/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\007\000\000\000\007\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
        dim {
          size: 7
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv4/kernel/RMSProp_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/kernel"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/Initializer/ones"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/Initializer/ones/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/Assign"
  op: "Assign"
  input: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp"
  input: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp/read"
  op: "Identity"
  input: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp_1/Initializer/zeros"
  op: "Fill"
  input: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp_1/Initializer/zeros/shape_as_tensor"
  input: "train/OptimizeLoss/stage_3/mid_conv4/bias/RMSProp_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@stage_3/mid_conv4/bias"
      }
    }
  }
  attr {
}
"""Convolutional Layer implementation."""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import conv_utils

# Conv2dTranspose Imports
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.eager import context
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

import tf_encrypted as tfe
from tf_encrypted.keras import activations
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras.engine import Layer
from tf_encrypted.keras.layers.layers_utils import default_args_check
from tf_encrypted.protocol.pond import PondPrivateTensor

logger = logging.getLogger("tf_encrypted")


class Conv2D(Layer):
    """2D convolution layer (e.g. spatial convolution over images).
  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.
  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          height and width of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the height and width.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
  Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
  """

    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            data_format=None,
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs,
    ):

        super(Conv2D, self).__init__(**kwargs)

        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, self.rank, "kernel_size"
        )
        if self.kernel_size[0] != self.kernel_size[1]:
            raise NotImplementedError(
                "TF Encrypted currently only supports same "
                "stride along the height and the width."
                "You gave: {}".format(self.kernel_size)
            )
        self.strides = conv_utils.normalize_tuple(strides, self.rank, "strides")
        self.padding = conv_utils.normalize_padding(padding).upper()
        self.data_format = conv_utils.normalize_data_format(data_format)
        if activation is not None:
            logger.info(
                "Performing an activation before a pooling layer can result "
                "in unnecessary performance loss. Check model definition in "
                "case of missed optimization."
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Not implemented arguments
        default_args_check(dilation_rate, "dilation_rate", "Conv2D")
        default_args_check(kernel_regularizer, "kernel_regularizer", "Conv2D")
        default_args_check(bias_regularizer, "bias_regularizer", "Conv2D")
        default_args_check(activity_regularizer, "activity_regularizer", "Conv2D")
        default_args_check(kernel_constraint, "kernel_constraint", "Conv2D")
        default_args_check(bias_constraint, "bias_constraint", "Conv2D")

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        input_dim = int(input_shape[channel_axis])
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)

        kernel = self.kernel_initializer(self.kernel_shape)
        self.kernel = self.add_weight(kernel)

        if self.use_bias:
            # Expand bias shape dimensions. Bias needs to have
            # a rank of 3 to be added to the output
            bias_shape = [self.filters, 1, 1]
            bias = self.bias_initializer(bias_shape)
            self.bias = self.add_weight(bias)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):

        if self.data_format != "channels_first":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])

        outputs = tfe.conv2d(inputs, self.kernel, self.strides[0], self.padding)

        if self.use_bias:
            outputs = outputs + self.bias

        if self.data_format != "channels_first":
            outputs = tfe.transpose(outputs, perm=[0, 2, 3, 1])

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        """Compute output_shape for the layer."""
        h_filter, w_filter, _, n_filters = self.kernel_shape

        if self.data_format == "channels_first":
            n_x, _, h_x, w_x = input_shape.as_list()
        else:
            n_x, h_x, w_x, _ = input_shape.as_list()

        if self.padding == "SAME":
            h_out = int(np.ceil(float(h_x) / float(self.strides[0])))
            w_out = int(np.ceil(float(w_x) / float(self.strides[0])))
        if self.padding == "VALID":
            h_out = int(np.ceil(float(h_x - h_filter + 1) / float(self.strides[0])))
            w_out = int(np.ceil(float(w_x - w_filter + 1) / float(self.strides[0])))

        return [n_x, n_filters, h_out, w_out]


class DepthwiseConv2D(Conv2D):
    """Depthwise separable 2D convolution.

  Depthwise Separable convolutions consists in performing
  just the first step in a depthwise spatial convolution
  (which acts on each input channel separately).
  The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.

  Arguments:
    kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: one of `'valid'` or `'same'` (case-insensitive).
    depth_multiplier: The number of depthwise convolution output channels
        for each input channel.
        The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
    data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be 'channels_last'.
    activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. 'linear' activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix.
    bias_initializer: Initializer for the bias vector.
    depthwise_regularizer: Regularizer function applied to
        the depthwise kernel matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
        the output of the layer (its 'activation').
    depthwise_constraint: Constraint function applied to
        the depthwise kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    4D tensor with shape:
    `[batch, channels, rows, cols]` if data_format='channels_first'
    or 4D tensor with shape:
    `[batch, rows, cols, channels]` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
    or 4D tensor with shape:
    `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
  """

    def __init__(
            self,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            depth_multiplier=1,
            data_format=None,
            activation=None,
            use_bias=True,
            depthwise_initializer="glorot_uniform",
            bias_initializer="zeros",
            depthwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
            **kwargs,
    ):

        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, self.rank, "kernel_size"
        )
        if self.kernel_size[0] != self.kernel_size[1]:
            raise NotImplementedError(
                "TF Encrypted currently only supports same "
                "stride along the height and the width."
                "You gave: {}".format(self.kernel_size)
            )
        self.strides = conv_utils.normalize_tuple(strides, self.rank, "strides")
        self.padding = conv_utils.normalize_padding(padding).upper()
        self.depth_multiplier = depth_multiplier
        self.data_format = conv_utils.normalize_data_format(data_format)
        if activation is not None:
            logger.info(
                "Performing an activation before a pooling layer can result "
                "in unnecessary performance loss. Check model definition in "
                "case of missed optimization."
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Not implemented arguments
        default_args_check(
            depthwise_regularizer, "depthwise_regularizer", "DepthwiseConv2D",
        )
        default_args_check(
            bias_regularizer, "bias_regularizer", "DepthwiseConv2D",
        )
        default_args_check(
            activity_regularizer, "activity_regularizer", "DepthwiseConv2D",
        )
        default_args_check(
            depthwise_constraint, "depthwise_constraint", "DepthwiseConv2D",
        )
        default_args_check(
            bias_constraint, "bias_constraint", "DepthwiseConv2D",
        )

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        self.input_dim = int(input_shape[channel_axis])
        self.kernel_shape = self.kernel_size + (self.input_dim, self.depth_multiplier)

        kernel = self.depthwise_initializer(self.kernel_shape)
        kernel = self.rearrange_kernel(kernel)
        self.kernel = self.add_weight(kernel)

        if self.use_bias:
            # Expand bias shape dimensions. Bias needs to have
            # a rank of 3 to be added to the output
            bias_shape = [self.input_dim * self.depth_multiplier, 1, 1]
            bias = self.bias_initializer(bias_shape)
            self.bias = self.add_weight(bias)
        else:
            self.bias = None

        self.built = True

    def rearrange_kernel(self, kernel):
        """Rearrange kernel to match normal convolution kernels

    Arguments:
      kernel: kernel to be rearranged
    """
        mask = self.get_mask(self.input_dim)

        if isinstance(kernel, tf.Tensor):
            mask = tf.constant(
                mask.tolist(),
                dtype=tf.float32,
                shape=(
                    self.kernel_size[0],
                    self.kernel_size[1],
                    self.input_dim * self.depth_multiplier,
                    self.input_dim,
                ),
            )

            if self.depth_multiplier > 1:
                # rearrange kernel
                kernel = tf.transpose(kernel, [0, 1, 3, 2])
                kernel = tf.reshape(
                    kernel,
                    shape=self.kernel_size
                          + (self.input_dim * self.depth_multiplier, 1),
                )

            kernel = tf.multiply(kernel, mask)

        elif isinstance(kernel, np.ndarray):
            if self.depth_multiplier > 1:
                # rearrange kernel
                kernel = np.transpose(kernel, [0, 1, 3, 2])
                kernel = np.reshape(
                    kernel,
                    newshape=self.kernel_size
                             + (self.input_dim * self.depth_multiplier, 1),
                )

            kernel = np.multiply(kernel, mask)

        elif isinstance(kernel, PondPrivateTensor):
            mask = tfe.define_public_variable(mask)
            if self.depth_multiplier > 1:
                # rearrange kernel
                kernel = tfe.transpose(kernel, [0, 1, 3, 2])
                kernel = tfe.reshape(
                    kernel,
                    shape=self.kernel_size
                          + (self.input_dim * self.depth_multiplier, 1),
                )

            kernel = tfe.mul(kernel, mask)

        return kernel

    def call(self, inputs):

        if self.data_format != "channels_first":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])

        outputs = tfe.conv2d(inputs, self.kernel, self.strides[0], self.padding)

        if self.use_bias:
            outputs = outputs + self.bias

        if self.data_format != "channels_first":
            outputs = tfe.transpose(outputs, perm=[0, 2, 3, 1])

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        """Compute output_shape for the layer."""
        h_filter, w_filter, _, n_filters = self.kernel_shape

        if self.data_format == "channels_first":
            n_x, _, h_x, w_x = input_shape.as_list()
        else:
            n_x, h_x, w_x, _ = input_shape.as_list()

        if self.padding == "SAME":
            h_out = int(np.ceil(float(h_x) / float(self.strides[0])))
            w_out = int(np.ceil(float(w_x) / float(self.strides[0])))
        if self.padding == "VALID":
            h_out = int(np.ceil(float(h_x - h_filter + 1) / float(self.strides[0])))
            w_out = int(np.ceil(float(w_x - w_filter + 1) / float(self.strides[0])))

        return [n_x, n_filters, h_out, w_out]

    def get_mask(self, in_channels):
        """TODO"""
        mask = np.zeros(
            (
                self.kernel_size[0],
                self.kernel_size[1],
                in_channels,
                in_channels * self.depth_multiplier,
            )
        )
        for d in range(self.depth_multiplier):
            for i in range(in_channels):
                mask[:, :, i, i + (d * in_channels)] = 1.0
        return np.transpose(mask, [0, 1, 3, 2])

    def set_weights(self, weights, sess=None):
        """
    Sets the weights of the layer.

    Arguments:
      weights: A list of Numpy arrays with shapes and types
          matching the output of layer.get_weights() or a list
          of private variables
      sess: tfe session"""

        weights_types = (np.ndarray, PondPrivateTensor)
        assert isinstance(weights[0], weights_types), type(weights[0])

        # Assign new keras weights to existing weights defined by
        # default when tfe layer was instantiated
        if not sess:
            sess = KE.get_session()

        if isinstance(weights[0], np.ndarray):
            for i, w in enumerate(self.weights):
                shape = w.shape.as_list()
                tfe_weights_pl = tfe.define_private_placeholder(shape)

                new_weight = weights[i]
                if i == 0:
                    # kernel
                    new_weight = self.rearrange_kernel(new_weight)
                else:
                    # bias
                    new_weight = new_weight.reshape(shape)

                fd = tfe_weights_pl.feed(new_weight)
                sess.run(tfe.assign(w, tfe_weights_pl), feed_dict=fd)

        elif isinstance(weights[0], PondPrivateTensor):
            for i, w in enumerate(self.weights):
                shape = w.shape.as_list()

                new_weight = weights[i]
                if i == 0:
                    # kernel
                    new_weight = self.rearrange_kernel(new_weight)
                else:
                    # bias
                    new_weight = new_weight.reshape(shape)

                sess.run(tfe.assign(w, new_weight))


class Conv2DTranspose(Conv2D):
    """Transposed convolution layer (sometimes called Deconvolution).
  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.
  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    output_padding: An integer or tuple/list of 2 integers,
      specifying the amount of padding along the height and width
      of the output tensor.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).
  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified:
    ```
    new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    ```
  Returns:
    A tensor of rank 4 representing
    `activation(conv2dtranspose(inputs, kernel) + bias)`.
  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            # kernel_regularizer=regularizers.get(kernel_regularizer),
            # bias_regularizer=regularizers.get(bias_regularizer),
            # activity_regularizer=regularizers.get(activity_regularizer),
            # kernel_constraint=constraints.get(kernel_constraint),
            # bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        # Not implemented arguments
        default_args_check(dilation_rate, "dilation_rate", "Conv2DTranspose")
        default_args_check(kernel_regularizer, "kernel_regularizer", "Conv2DTranspose")
        default_args_check(bias_regularizer, "bias_regularizer", "Conv2DTranspose")
        default_args_check(activity_regularizer, "activity_regularizer", "Conv2DTranspose")
        default_args_check(kernel_constraint, "kernel_constraint", "Conv2DTranspose")
        default_args_check(bias_constraint, "bias_constraint", "Conv2DTranspose")

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                                                     'greater than output padding ' +
                                     str(self.output_padding))

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input shape: ' +
                             str(input_shape))
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)


        # outputs = backend.conv2d_transpose(
        #     inputs,
        #     self.kernel,
        #     output_shape_tensor,
        #     strides=self.strides,
        #     padding=self.padding,
        #     data_format=self.data_format,
        #     dilation_rate=self.dilation_rate)


        # TODO: Calculate new padding input
        # TODO: Check whether padding is VALID or SAME are valid
        # TODO: Check whether spacing is needed
        # TODO: Check whether additional space is needed a = (i + 2p - k) mod s bottom and right edges
        # TODO: Apply standard convolution operation




        tfe.conv2d(inputs, self.kernel, self.strides[0], self.padding)

        outputs = tfe.conv2d()

        # return gen_nn_ops.conv2d_backprop_input(
        #     input_sizes=output_shape,
        #     filter=filters,
        #     out_backprop=input,
        #     strides=strides,
        #     padding=padding,
        #     data_format=data_format,
        #     dilations=dilations,
        #     name=name)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0])
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = super(Conv2DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        return config


# @tf_export("nn.conv2d_transpose", v1=[])
# def conv2d_transpose_v2(
#     input,  # pylint: disable=redefined-builtin
#     filters,  # pylint: disable=redefined-builtin
#     output_shape,
#     strides,
#     padding="SAME",
#     data_format="NHWC",
#     dilations=None,
#     name=None):
#   """The transpose of `conv2d`.
#   This operation is sometimes called "deconvolution" after
#   (Zeiler et al., 2010), but is really the transpose (gradient) of
#   `atrous_conv2d` rather than an actual deconvolution.
#   Args:
#     input: A 4-D `Tensor` of type `float` and shape `[batch, height, width,
#       in_channels]` for `NHWC` data format or `[batch, in_channels, height,
#       width]` for `NCHW` data format.
#     filters: A 4-D `Tensor` with the same type as `input` and shape `[height,
#       width, output_channels, in_channels]`.  `filter`'s `in_channels` dimension
#       must match that of `input`.
#     output_shape: A 1-D `Tensor` representing the output shape of the
#       deconvolution op.
#     strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
#       stride of the sliding window for each dimension of `input`. If a single
#       value is given it is replicated in the `H` and `W` dimension. By default
#       the `N` and `C` dimensions are set to 0. The dimension order is determined
#       by the value of `data_format`, see below for details.
#     padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
#       the "returns" section of `tf.nn.convolution` for details.
#     data_format: A string. 'NHWC' and 'NCHW' are supported.
#     dilations: An int or list of `ints` that has length `1`, `2` or `4`,
#       defaults to 1. The dilation factor for each dimension of`input`. If a
#       single value is given it is replicated in the `H` and `W` dimension. By
#       default the `N` and `C` dimensions are set to 1. If set to k > 1, there
#       will be k-1 skipped cells between each filter element on that dimension.
#       The dimension order is determined by the value of `data_format`, see above
#       for details. Dilations in the batch and depth dimensions if a 4-d tensor
#       must be 1.
#     name: Optional name for the returned tensor.
#   Returns:
#     A `Tensor` with the same type as `input`.
#   Raises:
#     ValueError: If input/output depth does not match `filter`'s shape, or if
#       padding is other than `'VALID'` or `'SAME'`.
#   References:
#     Deconvolutional Networks:
#       [Zeiler et al., 2010]
#       (https://ieeexplore.ieee.org/abstract/document/5539957)
#       ([pdf]
#       (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
#   """
#   with ops.name_scope(name, "conv2d_transpose",
#                       [input, filter, output_shape]) as name:
#     if data_format is None:
#       data_format = "NHWC"
#     channel_index = 1 if data_format.startswith("NC") else 3
#
#     strides = _get_sequence(strides, 2, channel_index, "strides")
#     dilations = _get_sequence(dilations, 2, channel_index, "dilations")
#
#     return gen_nn_ops.conv2d_backprop_input(
#         input_sizes=output_shape,
#         filter=filters,
#         out_backprop=input,
#         strides=strides,
#         padding=padding,
#         data_format=data_format,
#         dilations=dilations,
#         name=name)

# @tf_export(v1=["nn.conv2d_backprop_input"])
# def conv2d_backprop_input(  # pylint: disable=redefined-builtin,dangerous-default-value
#     input_sizes,
#     filter=None,
#     out_backprop=None,
#     strides=None,
#     padding=None,
#     use_cudnn_on_gpu=True,
#     data_format="NHWC",
#     dilations=[1, 1, 1, 1],
#     name=None,
#     filters=None):
#   r"""Computes the gradients of convolution with respect to the input.
#   Args:
#     input_sizes: A `Tensor` of type `int32`.
#       An integer vector representing the shape of `input`,
#       where `input` is a 4-D `[batch, height, width, channels]` tensor.
#     filter: A `Tensor`. Must be one of the following types:
#       `half`, `bfloat16`, `float32`, `float64`.
#       4-D with shape
#       `[filter_height, filter_width, in_channels, out_channels]`.
#     out_backprop: A `Tensor`. Must have the same type as `filter`.
#       4-D with shape `[batch, out_height, out_width, out_channels]`.
#       Gradients w.r.t. the output of the convolution.
#     strides: A list of `ints`.
#       The stride of the sliding window for each dimension of the input
#       of the convolution. Must be in the same order as the dimension specified
#       with format.
#     padding: Either the `string `"SAME"` or `"VALID"` indicating the type of
#       padding algorithm to use, or a list indicating the explicit paddings at
#       the start and end of each dimension. When explicit padding is used and
#       data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
#       pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
#       and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
#       [pad_top, pad_bottom], [pad_left, pad_right]]`.
#     use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
#     data_format: An optional `string` from: `"NHWC", "NCHW"`.
#       Defaults to `"NHWC"`.
#       Specify the data format of the input and output data. With the
#       default format "NHWC", the data is stored in the order of:
#           [batch, in_height, in_width, in_channels].
#       Alternatively, the format could be "NCHW", the data storage order of:
#           [batch, in_channels, in_height, in_width].
#     dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
#       1-D tensor of length 4.  The dilation factor for each dimension of
#       `input`. If set to k > 1, there will be k-1 skipped cells between each
#       filter element on that dimension. The dimension order is determined by
#       the value of `data_format`, see above for details. Dilations in the batch
#       and depth dimensions must be 1.
#     name: A name for the operation (optional).
#     filters: Alias for filter.
#   Returns:
#     A `Tensor`. Has the same type as `filter`.
#   """
#   filter = deprecation.deprecated_argument_lookup(
#       "filters", filters, "filter", filter)
#   padding, explicit_paddings = _convert_padding(padding)
#   return gen_nn_ops.conv2d_backprop_input(
#       input_sizes, filter, out_backprop, strides, padding, use_cudnn_on_gpu,
#       explicit_paddings, data_format, dilations, name)


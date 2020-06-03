from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.utils as conv_utils
import tensorflow as tf
import tensorflow.keras.backend as K

def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.
    # Arguments
        value: The value to validate and convert. Could be an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. `strides` or
          `kernel_size`. This is only used to format error messages.
    # Returns
        A tuple of n integers.
    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `{}` argument must be a tuple of {} '
                             'integers. Received: {}'.format(name, n, value))
        if len(value_tuple) != n:
            raise ValueError('The `{}` argument must be a tuple of {} '
                             'integers. Received: {}'.format(name, n, value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `{}` argument must be a tuple of {} '
                                 'integers. Received: {} including element {} '
                                 'of type {}'.format(name, n, value, single_value,
                                                     type(single_value)))
    return value_tuple


def normalize_data_format(value):
    """Checks that the value correspond to a valid data format.
    Copy of the function in keras-team/keras because it's not public API.
    # Arguments
        value: String or None. `'channels_first'` or `'channels_last'`.
    # Returns
        A string, either `'channels_first'` or `'channels_last'`
    # Example
    ```python
        >>> from keras import backend as K
        >>> K.normalize_data_format(None)
        'channels_first'
        >>> K.normalize_data_format('channels_last')
        'channels_last'
    ```
    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        return tf.compat.v1.image.resize(images=inputs, size=[height, width], align_corners=True)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

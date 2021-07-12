from typing import Optional, Sequence

# The point of this one is to remove ALL the batchnorms.


import larq as lq
import numpy as np
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory

from customRenorm import ReluNormalRenorm, MaxpoolRenorm#, BlurpoolRenorm

from tensorflow.keras import initializers

@factory
class RemadeQuickNetFactory(ModelFactory):
    name = "quicknet"
    section_blocks: Sequence[int] = Field((4, 4, 4, 4))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0",
            file="quicknet_weights.h5",
            file_hash="8aba9e4e5f8d342faef04a0b2ae8e562da57dbb7d15162e8b3e091c951ba756c",# Not correct filepath; from original QuickNet
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0",
            file="quicknet_weights_notop.h5",
            file_hash="204414e438373f14f6056a1c098249f505a87dd238e18d3a47a9bd8b66227881",# Not correct filepath; from original QuickNet
        )

    @property
    def input_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_constraint(self):
        return lq.constraints.WeightClip(clip_value=1.25)

    def __post_configure__(self):
        assert len(self.section_blocks) == len(self.section_filters)

    def stem_module(self, filters: int, x: tf.Tensor) -> tf.Tensor:
        """Start of network."""
        assert filters % 4 == 0
        # mu = 0, sigma = 1, if the preprocessor did its job
        x = lq.layers.QuantConv2D(
            filters // 4,
            (3, 3),
            kernel_initializer=initializers.RandomNormal(stddev=(1/np.sqrt(27))), # 27 = fan_in for 3-channel image and 3x3 convolutions
            padding="same",
            strides=2,
            use_bias=False,
        )(x)
        # mu = 0, sigma = 1.0, roughly normal
        x = tf.keras.layers.Activation("relu")(x)
        # Now, we need a correction for the relu 
        x = ReluNormalRenorm(N = 1, origSig = 1.0)(x)
        # mu = 0, sigma = 1, NOT normal
        x = lq.layers.QuantDepthwiseConv2D(
            (3, 3),
            depthwise_initializer=initializers.RandomNormal(stddev=(1.0/3)), # 9 = fan_in for 3x3 depthwise convolution, sqrt(9) = 3
            padding="same",
            strides=2,
            use_bias=False,
        )(x) 
        # mu = 0, sigma = 1.0, roughly normal
        x = lq.layers.QuantConv2D(
            filters,
            1,
            kernel_initializer=initializers.RandomNormal(stddev=(1/(np.sqrt(filters // 4)))), # It's a 1x1 convolution, fan_in = input_channels = filters // 4
            use_bias=False,
        )(x)
        # mu = 0, sigma = 1.0, roughly normal
        return x # mu = 0, sigma = 1

    def residual_block(self, x: tf.Tensor) -> tf.Tensor:
        """Standard residual block, without strides or filter changes."""
        # mu = 0, sig = origSig
        residual = x
        x = lq.layers.QuantConv2D(
            int(x.shape[-1]),
            (3, 3),
            activation="relu",
            input_quantizer=self.input_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer="glorot_normal",
            padding="same",
            pad_values=1.0,
            use_bias=False,
        )(x)
        x = ReluNormalRenorm(N = 3*3*int(x.shape[-1]))(x)
        # mu = 0, sigma = 1, NOT normal
        return x + residual # mu = 0, sig = sqrt(1 + origSig**2). At the end of 3 blocks, its sqrt(4) = 2.0. NOT normal

    def transition_block(
        self,
        x: tf.Tensor,
        filters: int,
        strides: int,
        prev_blocks: int,
    ) -> tf.Tensor:
        """Pointwise transition block."""
        # Input is after 3 residual blocks, so mu = 0, sig = np.sqrt(prev_blocks). NOT normal.
        x = lq.layers.QuantDepthwiseConv2D(
            (3, 3),
            depthwise_initializer=initializers.RandomNormal(stddev=(1.0/(3*np.sqrt(prev_blocks + 1)))), # fan_in for 3x3 depthwiseConv is 9, 9^0.5 = 3. Divide out initial sigma
            padding="same",
            strides=2,
            use_bias=False,
        )(x)
        # mu = 0, sig = 1, roughly normal. Must be normal for MaxpoolRenorm to work correctly.
        x = tf.keras.layers.MaxPool2D(pool_size=strides, strides=strides)(x)
        x = MaxpoolRenorm(N = 1, M = strides*strides)(x) # N is always 1 for well-initialized full-precision layers.
        # mu = 0, sig = 1, NOT normal
        x = lq.layers.QuantConv2D(
            filters,
            (1, 1),
            kernel_initializer=initializers.RandomNormal(stddev=(1/np.sqrt(x.shape[-1]))),
            use_bias=False,
        )(x)
        # Both initialized correctly, should have mu = 0, sig = 1, roughly normal.
        return x

    def build(self) -> tf.keras.models.Model:
        x = self.stem_module(self.section_filters[0], self.image_input)
        prev_blocks = 0
        for block, (layers, filters) in enumerate(
            zip(self.section_blocks, self.section_filters)
        ):
            for layer in range(layers):
                if filters != x.shape[-1]:
                    x = self.transition_block(x, filters, strides=2, prev_blocks=prev_blocks)
                    prev_blocks = 0
                x = self.residual_block(x)
                prev_blocks += 1

        if self.include_top:
            x = tf.keras.layers.Activation("relu")(x)
            x = utils.global_pool(x)
            x = lq.layers.QuantDense(
                self.num_classes,
                kernel_initializer="glorot_normal",
            )(x)
            x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=self.image_input, outputs=x, name=self.name)

        # Load weights.
        if self.weights == "imagenet":
            weights_path = (
                self.imagenet_weights_path
                if self.include_top
                else self.imagenet_no_top_weights_path
            )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


@factory
class RemadeQuickNetSmallFactory(RemadeQuickNetFactory):
    name = "quicknet_small"
    section_filters = Field((32, 64, 256, 512))

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0",
            file="quicknet_small_weights.h5",
            file_hash="1ac3b07df7f5a911dd0b49febb2486428ddf1ca130297c403815dfae5a1c71a2",# Not correct filepath; from original QuickNet
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0",
            file="quicknet_small_weights_notop.h5",
            file_hash="be8ba657155846be355c5580d1ea56eaf8282616de065ffc39257202f9f164ea",# Not correct filepath; from original QuickNet
        )


@factory
class RemadeQuickNetLargeFactory(RemadeQuickNetFactory):
    name = "quicknet_large"
    section_blocks = Field((6, 8, 12, 6))

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0",
            file="quicknet_large_weights.h5",
            file_hash="c5158e8a59147b31370becd937825f4db8a5cdf308314874f678f596629be45c",# Not correct filepath; from original QuickNet
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0",
            file="quicknet_large_weights_notop.h5",
            file_hash="adcf154a2a8007e81bd6af77c035ffbf54cd6413b89a0ba294e23e76a82a1b78",# Not correct filepath; from original QuickNet
        )


def RemadeQuickNet(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the RemadeQuickNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    ```netron
    quicknet-v1.0/quicknet.json
    ```
    ```summary
    sota.RemadeQuickNet
    ```
    ```plot-altair
    /plots/quicknet.vg.json
    ```
    # ImageNet Metrics
    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 63.3 %         | 84.6 %         | 13 234 088 | 4.17 MB |
    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        weights: one of `None` (random initialization), "imagenet" (pre-training on
            ImageNet), or the path to the weights file to be loaded.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.
    """
    return RemadeQuickNetFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


def RemadeQuickNetLarge(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the RemadeQuickNetLarge architecture.
    Optionally loads weights pre-trained on ImageNet.
    ```netron
    quicknet-v1.0/quicknet_large.json
    ```
    ```summary
    sota.RemadeQuickNetLarge
    ```
    ```plot-altair
    /plots/quicknet_large.vg.json
    ```
    # ImageNet Metrics
    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 66.9 %         | 87.0 %         | 23 342 248 | 5.40 MB |
    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        weights: one of `None` (random initialization), "imagenet" (pre-training on
            ImageNet), or the path to the weights file to be loaded.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.
    """
    return RemadeQuickNetLargeFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


def RemadeQuickNetSmall(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the RemadeQuickNetSmall architecture.
    Optionally loads weights pre-trained on ImageNet.
    ```netron
    quicknet-v1.0/quicknet_small.json
    ```
    ```summary
    sota.RemadeQuickNetSmall
    ```
    ```plot-altair
    /plots/quicknet_small.vg.json
    ```
    # ImageNet Metrics
    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 59.4 %         | 81.8 %         | 12 655 688 | 4.00 MB |
    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        weights: one of `None` (random initialization), "imagenet" (pre-training on
            ImageNet), or the path to the weights file to be loaded.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.
    """
    return RemadeQuickNetSmallFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


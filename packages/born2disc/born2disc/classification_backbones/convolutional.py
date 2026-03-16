import math
import torch as pt

from dataclasses import field
from typing import Iterable, List, Optional, Tuple, Union, Literal

from nestconf import Configurable, ConfigurableABCMeta

from ..lightning_modules.feedforward_classification import ClassificationBackbone


def _normalize_sequence(value: Union[Iterable[int], int], length: int, name: str) -> List[int]:
    if isinstance(value, int):
        return [value] * length
    value = list(value)
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {len(value)}")
    return value


def _make_activation(
    *,
    activation_ctor: type[pt.nn.Module],
) -> pt.nn.Module:
    return activation_ctor()


# -------------------------
# Normalization factories
# -------------------------

def _make_norm2d(
    *,
    norm_type: Literal["none", "batch", "group"],
    num_channels: int,
    gn_num_groups: int,
) -> pt.nn.Module:
    if norm_type == "none":
        return pt.nn.Identity()

    if norm_type == "batch":
        return pt.nn.BatchNorm2d(num_features=num_channels)

    if norm_type == "group":
        num_groups = min(gn_num_groups, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return pt.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
        )

    raise ValueError(f"Unknown norm_type={norm_type!r}.")


def _make_norm1d(
    *,
    norm_type: Literal["none", "batch", "layer"],
    num_features: int,
) -> pt.nn.Module:
    if norm_type == "none":
        return pt.nn.Identity()

    if norm_type == "batch":
        return pt.nn.BatchNorm1d(num_features=num_features)

    if norm_type == "layer":
        return pt.nn.LayerNorm(normalized_shape=num_features)

    raise ValueError(f"Unknown norm_type={norm_type!r}.")


# -------------------------
# Linear block (MLP / residual)
# -------------------------

def _make_linear_block(
    *,
    in_dim: int,
    out_dim: int,
    activation_ctor: type[pt.nn.Module],
    dropout: Optional[float],
    norm_type: Literal["none", "batch", "layer"],
) -> pt.nn.Sequential:
    layers: List[pt.nn.Module] = [
        pt.nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=(norm_type == "none"),
        ),
        _make_norm1d(
            norm_type=norm_type,
            num_features=out_dim,
        ),
        _make_activation(
            activation_ctor=activation_ctor,
        ),
    ]
    if dropout is not None:
        layers.append(
            pt.nn.Dropout(p=dropout),
        )
    return pt.nn.Sequential(*layers)

# -------------------------
# Feature engineering layers
# -------------------------

class _XYBondEnergyFeatures(pt.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        # x: (B, 2, L, L) with channels (cos, sin)
        x_right = pt.roll(
            x,
            shifts=-1,
            dims=3,
        )
        x_down = pt.roll(
            x,
            shifts=-1,
            dims=2,
        )

        e_right = pt.sum(
            x * x_right,
            dim=1,
            keepdim=True,
        )
        e_down = pt.sum(
            x * x_down,
            dim=1,
            keepdim=True,
        )

        # (B, 2, L, L): [e_right, e_down]
        return pt.cat(
            [e_right, e_down],
            dim=1,
        )

# -------------------------
# Convolutional block
# -------------------------

class _ConvBlock(pt.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_type: Literal["none", "batch", "group"],
        gn_num_groups: int,
        dropout: float,
        activation_ctor: type[pt.nn.Module],
    ) -> None:
        super().__init__()

        use_bias = norm_type == "none"

        self.conv = pt.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="circular",
            bias=use_bias,
        )

        self.norm = _make_norm2d(
            norm_type=norm_type,
            num_channels=out_channels,
            gn_num_groups=gn_num_groups,
        )

        self.act = _make_activation(activation_ctor=activation_ctor)
        self.dropout = pt.nn.Dropout2d(p=dropout) if dropout > 0.0 else pt.nn.Identity()

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


# -------------------------
# Backbone
# -------------------------

class ConvolutionalClassificationBackbone(
    ClassificationBackbone, Configurable, metaclass=ConfigurableABCMeta
):
    conv_channels: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    conv_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    conv_paddings: List[int] = field(default_factory=lambda: [1, 1, 1, 1])

    conv_norm_type: Literal["none", "batch", "group"] = "batch"
    conv_gn_num_groups: int = 8

    conv_dropout: float = 0.0
    conv_activation_ctor: type[pt.nn.Module] = pt.nn.LeakyReLU

    hidden_size: int = 64
    num_hidden_layers: int = 1

    # NEW: LayerNorm option for MLP/residual trunk
    ff_norm_type: Literal["none", "batch", "layer"] = "layer"

    activation_ctor: type[pt.nn.Module] = pt.nn.Tanh
    use_residual_connections: bool = True
    dropout: float = 0.0

    input_channels: int = 1
    input_shape: Optional[Tuple[int, int]] = None
    basis_label_included: bool = True
    basis_label_size: int = 1

    add_xy_bond_features: bool = True

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int = 8,
        **kwargs,
    ):
        super().__init__(embedding_size=embedding_size, **kwargs)

        self.input_size = input_size
        self.embedding_size = embedding_size

        image_input_size = input_size
        if self.basis_label_included:
            if input_size <= self.basis_label_size:
                raise ValueError("input_size must be larger than basis_label_size")
            image_input_size -= self.basis_label_size
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")

        if self.input_shape is None:
            if image_input_size % self.input_channels != 0:
                raise ValueError(
                    f"image_input_size {image_input_size} is not divisible by "
                    f"input_channels {self.input_channels}"
                )
            spatial = image_input_size // self.input_channels
            side = int(math.isqrt(spatial))
            if side * side != spatial:
                raise ValueError(
                    "input_shape must be provided when input_size is not a square image size "
                    f"(input_size={input_size}, basis_label_included={self.basis_label_included}, "
                    f"input_channels={self.input_channels})."
                )
            self.input_shape = (side, side)
        else:
            expected = self.input_channels * self.input_shape[0] * self.input_shape[1]
            if expected != image_input_size:
                raise ValueError(
                    f"input_shape {self.input_shape} and input_channels {self.input_channels} "
                    f"do not match input_size {input_size} (basis_label_included={self.basis_label_included})"
                )
            
        self.xy_bond_features = _XYBondEnergyFeatures() if self.add_xy_bond_features else None
        extra_channels = 2 if self.add_xy_bond_features else 0

        num_conv_layers = len(self.conv_channels)
        kernel_sizes = _normalize_sequence(self.conv_kernel_sizes, num_conv_layers, "conv_kernel_sizes")
        strides = _normalize_sequence(self.conv_strides, num_conv_layers, "conv_strides")
        paddings = _normalize_sequence(self.conv_paddings, num_conv_layers, "conv_paddings")

        conv_layers: List[pt.nn.Module] = []
        in_channels = self.input_channels + extra_channels
        for out_channels, k, s, p in zip(self.conv_channels, kernel_sizes, strides, paddings):
            conv_layers.append(
                _ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    norm_type=self.conv_norm_type,
                    gn_num_groups=self.conv_gn_num_groups,
                    dropout=self.conv_dropout,
                    activation_ctor=self.conv_activation_ctor,
                )
            )
            in_channels = out_channels

        self.conv_layers = pt.nn.Sequential(*conv_layers)
        self.conv_pool = pt.nn.AdaptiveAvgPool2d((1, 1))

        conv_output_dim = self.conv_channels[-1]
        ff_input_dim = conv_output_dim + (self.basis_label_size if self.basis_label_included else 0)
        dropout = self.dropout if self.dropout > 0.0 else None

        if self.num_hidden_layers > 0:
            self.input_layer = _make_linear_block(
                in_dim=ff_input_dim,
                out_dim=self.hidden_size,
                activation_ctor=self.activation_ctor,
                dropout=dropout,
                norm_type=self.ff_norm_type,
            )

            self.hidden_layers = pt.nn.Sequential(
                *[
                    _make_linear_block(
                        in_dim=self.hidden_size,
                        out_dim=self.hidden_size,
                        activation_ctor=self.activation_ctor,
                        dropout=dropout,
                        norm_type=self.ff_norm_type,
                    )
                    for _ in range(self.num_hidden_layers)
                ]
            )

            # IMPORTANT: no normalization on embedding layer
            self.embedding_layer = _make_linear_block(
                in_dim=self.hidden_size,
                out_dim=self.embedding_size,
                activation_ctor=self.activation_ctor,
                dropout=dropout,
                norm_type="none",
            )
        else:
            self.embedding_layer = _make_linear_block(
                in_dim=ff_input_dim,
                out_dim=self.embedding_size,
                activation_ctor=self.activation_ctor,
                dropout=dropout,
                norm_type="none",
            )

    def forward(self, x: pt.Tensor, dtype: pt.dtype = None) -> pt.Tensor:
        if self.basis_label_included:
            basis_label = x[:, : self.basis_label_size]
            x = x[:, self.basis_label_size :]
        else:
            basis_label = None

        x = x.reshape(
            x.shape[0],
            self.input_channels,
            self.input_shape[0],
            self.input_shape[1],
        )

        if self.xy_bond_features is not None:
            bond_features = self.xy_bond_features(x)
            x = pt.cat([x, bond_features], dim=1)

        x = self.conv_layers(x)
        x = self.conv_pool(x)
        x = x.reshape(x.shape[0], -1)

        if basis_label is not None:
            x = pt.cat([basis_label, x], dim=1)

        if self.num_hidden_layers > 0:
            x = self.input_layer(x)
            for layer in self.hidden_layers:
                x_in = x
                x = layer(x)
                if self.use_residual_connections:
                    x = x + x_in
            x = self.embedding_layer(x)
        else:
            x = self.embedding_layer(x)

        return x
# import math
# import torch as pt

# from dataclasses import field
# from typing import Iterable, List, Optional, Tuple, Union

# from nestconf import Configurable, ConfigurableABCMeta

# from ..lightning_modules.feedforward_classification import ClassificationBackbone


# def _normalize_sequence(value: Union[Iterable[int], int], length: int, name: str) -> List[int]:
#     if isinstance(value, int):
#         return [value] * length
#     value = list(value)
#     if len(value) != length:
#         raise ValueError(f"{name} must have length {length}, got {len(value)}")
#     return value


# def _make_activation(
#     *,
#     activation_ctor: type[pt.nn.Module],
# ) -> pt.nn.Module:
#     return activation_ctor()


# def _make_linear_block(
#     *,
#     in_dim: int,
#     out_dim: int,
#     activation_ctor: type[pt.nn.Module],
#     dropout: Optional[float],
# ) -> pt.nn.Sequential:
#     layers: List[pt.nn.Module] = [
#         pt.nn.Linear(in_features=in_dim, out_features=out_dim),
#     ]
#     if dropout is not None:
#         layers.append(pt.nn.Dropout(p=dropout))
#     layers.append(_make_activation(activation_ctor=activation_ctor))
#     return pt.nn.Sequential(*layers)


# def _make_groupnorm(
#     *,
#     num_channels: int,
#     use_groupnorm: bool,
#     gn_num_groups: int,
# ) -> pt.nn.Module:
#     if not use_groupnorm:
#         return pt.nn.Identity()
#     num_groups = min(gn_num_groups, num_channels)
#     while num_channels % num_groups != 0 and num_groups > 1:
#         num_groups -= 1
#     return pt.nn.GroupNorm(
#         num_groups=num_groups,
#         num_channels=num_channels,
#     )


# class ResidualBlock2d(pt.nn.Module):
#     """
#     A basic ResNet-style block:

#         y = Act( Skip(x) + Conv-Norm-Act-(Drop)-Conv-Norm(x) )

#     - If stride != 1 or in_channels != out_channels, Skip is a 1x1 conv projection.
#     - Norm is GroupNorm if enabled; otherwise Identity.
#     """

#     def __init__(
#         self,
#         *,
#         in_channels: int,
#         out_channels: int,
#         stride: int,
#         use_groupnorm: bool,
#         gn_num_groups: int,
#         dropout: float,
#         activation_ctor: type[pt.nn.Module],
#     ) -> None:
#         super().__init__()

#         self.conv1 = pt.nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=not use_groupnorm,
#         )
#         self.norm1 = _make_groupnorm(
#             num_channels=out_channels,
#             use_groupnorm=use_groupnorm,
#             gn_num_groups=gn_num_groups,
#         )
#         self.act1 = _make_activation(activation_ctor=activation_ctor)
#         self.drop = pt.nn.Dropout2d(p=dropout) if dropout > 0.0 else pt.nn.Identity()

#         self.conv2 = pt.nn.Conv2d(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=not use_groupnorm,
#         )
#         self.norm2 = _make_groupnorm(
#             num_channels=out_channels,
#             use_groupnorm=use_groupnorm,
#             gn_num_groups=gn_num_groups,
#         )

#         if stride != 1 or in_channels != out_channels:
#             self.skip = pt.nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=stride,
#                 padding=0,
#                 bias=False,
#             )
#         else:
#             self.skip = pt.nn.Identity()

#         self.act_out = _make_activation(activation_ctor=activation_ctor)

#     def forward(self, x: pt.Tensor) -> pt.Tensor:
#         identity = self.skip(x)

#         out = self.conv1(x)
#         out = self.norm1(out)
#         out = self.act1(out)
#         out = self.drop(out)

#         out = self.conv2(out)
#         out = self.norm2(out)

#         out = out + identity
#         out = self.act_out(out)
#         return out


# class ConvolutionalClassificationBackbone(ClassificationBackbone, Configurable, metaclass=ConfigurableABCMeta):
#     # These now define "stages" (one residual block per stage).
#     conv_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
#     conv_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])

#     # Residual blocks usually fix kernel to 3; keep padding implicit.
#     conv_use_groupnorm: bool = True
#     conv_gn_num_groups: int = 2
#     conv_dropout: float = 0.0
#     conv_activation_ctor: type[pt.nn.Module] = pt.nn.LeakyReLU

#     hidden_size: int = 128
#     num_hidden_layers: int = 1
#     activation_ctor: type[pt.nn.Module] = pt.nn.LeakyReLU
#     use_residual_connections: bool = True
#     dropout: float = 0.0

#     input_channels: int = 1
#     input_shape: Optional[Tuple[int, int]] = None
#     basis_label_included: bool = True
#     basis_label_size: int = 1

#     def __init__(
#         self,
#         *,
#         input_size: int,
#         embedding_size: int = 16,
#         **kwargs,
#     ):
#         super().__init__(embedding_size=embedding_size, **kwargs)

#         self.input_size = input_size
#         self.embedding_size = embedding_size

#         image_input_size = input_size
#         if self.basis_label_included:
#             if input_size <= self.basis_label_size:
#                 raise ValueError("input_size must be larger than basis_label_size")
#             image_input_size = input_size - self.basis_label_size

#         if self.input_shape is None:
#             spatial = image_input_size // self.input_channels
#             side = int(math.sqrt(spatial))
#             if side * side * self.input_channels != image_input_size:
#                 raise ValueError("input_shape must be provided when input_size is not a square image size")
#             self.input_shape = (side, side)
#         else:
#             expected = self.input_channels * self.input_shape[0] * self.input_shape[1]
#             if expected != image_input_size:
#                 raise ValueError(
#                     f"input_shape {self.input_shape} and input_channels {self.input_channels} "
#                     f"do not match input_size {input_size} (basis_label_included={self.basis_label_included})"
#                 )

#         num_stages = len(self.conv_channels)
#         strides = _normalize_sequence(self.conv_strides, num_stages, "conv_strides")

#         stages: List[pt.nn.Module] = []
#         in_channels = self.input_channels
#         for out_channels, stride in zip(self.conv_channels, strides):
#             stages.append(
#                 ResidualBlock2d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     stride=stride,
#                     use_groupnorm=self.conv_use_groupnorm,
#                     gn_num_groups=self.conv_gn_num_groups,
#                     dropout=self.conv_dropout,
#                     activation_ctor=self.conv_activation_ctor,
#                 )
#             )
#             in_channels = out_channels

#         self.conv_layers = pt.nn.Sequential(*stages)

#         self.conv_pool = pt.nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         conv_output_dim = self.conv_channels[-1]

#         ff_input_dim = conv_output_dim + (self.basis_label_size if self.basis_label_included else 0)
#         dropout = self.dropout if self.dropout > 0.0 else None

#         if self.num_hidden_layers > 0:
#             self.input_layer = _make_linear_block(
#                 in_dim=ff_input_dim,
#                 out_dim=self.hidden_size,
#                 activation_ctor=self.activation_ctor,
#                 dropout=dropout,
#             )
#             self.hidden_layers = pt.nn.Sequential(
#                 *[
#                     _make_linear_block(
#                         in_dim=self.hidden_size,
#                         out_dim=self.hidden_size,
#                         activation_ctor=self.activation_ctor,
#                         dropout=dropout,
#                     )
#                     for _ in range(self.num_hidden_layers)
#                 ]
#             )
#             self.embedding_layer = _make_linear_block(
#                 in_dim=self.hidden_size,
#                 out_dim=self.embedding_size,
#                 activation_ctor=self.activation_ctor,
#                 dropout=dropout,
#             )
#         else:
#             self.embedding_layer = _make_linear_block(
#                 in_dim=ff_input_dim,
#                 out_dim=self.embedding_size,
#                 activation_ctor=self.activation_ctor,
#                 dropout=dropout,
#             )

#     def forward(self, x: pt.Tensor, dtype: pt.dtype = None) -> pt.Tensor:
#         if self.basis_label_included:
#             basis_label = x[:, : self.basis_label_size]
#             x = x[:, self.basis_label_size :]
#         else:
#             basis_label = None

#         x = x.reshape(
#             x.shape[0],
#             self.input_channels,
#             self.input_shape[0],
#             self.input_shape[1],
#         )

#         x = self.conv_layers(x)
#         x = self.conv_pool(x)
#         x = x.reshape(x.shape[0], -1)

#         if basis_label is not None:
#             x = pt.cat([basis_label, x], dim=1)

#         if self.num_hidden_layers > 0:
#             x = self.input_layer(x)
#             for hidden_layer in self.hidden_layers:
#                 x_in = x
#                 x = hidden_layer(x)
#                 if self.use_residual_connections:
#                     x = x + x_in
#             x = self.embedding_layer(x)
#         else:
#             x = self.embedding_layer(x)

#         return x




# # import math
# # import torch as pt

# # from dataclasses import field
# # from typing import Iterable, List, Optional, Tuple, Union

# # from nestconf import Configurable, ConfigurableABCMeta

# # from ..lightning_modules.feedforward_classification import ClassificationBackbone


# # def _normalize_sequence(value: Union[Iterable[int], int], length: int, name: str) -> List[int]:
# #     if isinstance(value, int):
# #         return [value] * length
# #     value = list(value)
# #     if len(value) != length:
# #         raise ValueError(f"{name} must have length {length}, got {len(value)}")
# #     return value


# # def _make_linear_block(in_dim: int, out_dim: int, activation: pt.nn.Module, dropout: Optional[float]) -> pt.nn.Sequential:
# #     layers: List[pt.nn.Module] = [pt.nn.Linear(in_dim, out_dim)]
# #     if dropout is not None:
# #         layers.append(pt.nn.Dropout(dropout))
# #     layers.append(activation)
# #     return pt.nn.Sequential(*layers)


# # class ConvolutionalClassificationBackbone(ClassificationBackbone, Configurable, metaclass=ConfigurableABCMeta):
# #     conv_channels: List[int] = field(default_factory=lambda: [8, 8,])
# #     conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
# #     conv_strides: List[int] = field(default_factory=lambda: [1, 1])
# #     conv_paddings: List[int] = field(default_factory=lambda: [1, 1])
# #     conv_use_batchnorm: bool = False
# #     conv_dropout: float = 0.0
# #     conv_activation: pt.nn.Module = field(default_factory=pt.nn.LeakyReLU)

# #     hidden_size: int = 128
# #     num_hidden_layers: int = 1
# #     activation: pt.nn.Module = field(default_factory=pt.nn.LeakyReLU)
# #     use_residual_connections: bool = True
# #     dropout: float = 0.0

# #     input_channels: int = 1
# #     input_shape: Optional[Tuple[int, int]] = None
# #     basis_label_included: bool = True
# #     basis_label_size: int = 1

# #     def __init__(self,
# #                  *,
# #                  input_size: int = None,
# #                  embedding_size: int = 64,
# #                  **kwargs):
# #         super().__init__(embedding_size=embedding_size, **kwargs)
# #         if input_size is None:
# #             raise ValueError("input_size must be provided")

# #         self.input_size = input_size
# #         self.embedding_size = embedding_size

# #         image_input_size = input_size
# #         if self.basis_label_included:
# #             if input_size <= self.basis_label_size:
# #                 raise ValueError("input_size must be larger than basis_label_size")
# #             image_input_size = input_size - self.basis_label_size

# #         if self.input_shape is None:
# #             spatial = image_input_size // self.input_channels
# #             side = int(math.sqrt(spatial))
# #             if side * side * self.input_channels != image_input_size:
# #                 raise ValueError(
# #                     "input_shape must be provided when input_size is not a square image size"
# #                 )
# #             self.input_shape = (side, side)
# #         else:
# #             expected = self.input_channels * self.input_shape[0] * self.input_shape[1]
# #             if expected != image_input_size:
# #                 raise ValueError(
# #                     f"input_shape {self.input_shape} and input_channels {self.input_channels} "
# #                     f"do not match input_size {input_size} (basis_label_included={self.basis_label_included})"
# #                 )

# #         num_conv_layers = len(self.conv_channels)
# #         kernel_sizes = _normalize_sequence(self.conv_kernel_sizes, num_conv_layers, "conv_kernel_sizes")
# #         strides = _normalize_sequence(self.conv_strides, num_conv_layers, "conv_strides")
# #         paddings = _normalize_sequence(self.conv_paddings, num_conv_layers, "conv_paddings")

# #         conv_layers: List[pt.nn.Module] = []
# #         in_channels = self.input_channels
# #         for out_channels, kernel_size, stride, padding in zip(
# #             self.conv_channels, kernel_sizes, strides, paddings
# #         ):
# #             conv_layers.append(
# #                 pt.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
# #             )
# #             if self.conv_use_batchnorm:
# #                 conv_layers.append(pt.nn.BatchNorm2d(out_channels))
# #             if self.conv_dropout > 0.0:
# #                 conv_layers.append(pt.nn.Dropout2d(self.conv_dropout))
# #             conv_layers.append(self.conv_activation)
# #             in_channels = out_channels

# #         self.conv_layers = pt.nn.Sequential(*conv_layers)
# #        # self.conv_pool = pt.nn.AdaptiveAvgPool2d((1, 1))

# #         with pt.no_grad():
# #             dummy = pt.zeros(1, self.input_channels, self.input_shape[0], self.input_shape[1])
# #             conv_out = self.conv_layers(dummy)
# #             #conv_out = self.conv_pool(conv_out)
# #         conv_output_dim = int(conv_out.numel())

# #         ff_input_dim = conv_output_dim + (self.basis_label_size if self.basis_label_included else 0)
# #         dropout = self.dropout if self.dropout > 0.0 else None

# #         if self.num_hidden_layers > 0:
# #             self.input_layer = _make_linear_block(ff_input_dim, self.hidden_size, self.activation, dropout=dropout)
# #             self.hidden_layers = pt.nn.Sequential(
# #                 *[
# #                     _make_linear_block(self.hidden_size, self.hidden_size, self.activation, dropout=dropout)
# #                     for _ in range(self.num_hidden_layers)
# #                 ]
# #             )
# #             self.embedding_layer = _make_linear_block(self.hidden_size, self.embedding_size, self.activation, dropout=dropout)
# #         else:
# #             self.embedding_layer = _make_linear_block(ff_input_dim, self.embedding_size, self.activation, dropout=dropout)

# #     def forward(self, x: pt.Tensor, dtype: pt.dtype = None) -> pt.Tensor:
# #         if self.basis_label_included:
# #             basis_label = x[:, : self.basis_label_size]
# #             x = x[:, self.basis_label_size :]
# #         else:
# #             basis_label = None

# #         x = x.reshape(x.shape[0], self.input_channels, self.input_shape[0], self.input_shape[1])
# #         x = self.conv_layers(x)
# #         #x = self.conv_pool(x)
# #         x = x.reshape(x.shape[0], -1)

# #         if basis_label is not None:
# #             x = pt.cat([basis_label, x], dim=1)

# #         if self.num_hidden_layers > 0:
# #             x = self.input_layer(x)
# #             for hidden_layer in self.hidden_layers:
# #                 x_in = x
# #                 x = hidden_layer(x)
# #                 if self.use_residual_connections:
# #                     x = x + x_in
# #             x = self.embedding_layer(x)
# #         else:
# #             x = self.embedding_layer(x)

# #         return x

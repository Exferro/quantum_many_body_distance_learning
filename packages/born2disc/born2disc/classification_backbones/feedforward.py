# file: feedforward.py
import torch as pt

from dataclasses import field
from typing import List, Literal, Optional

from nestconf import Configurable, ConfigurableABCMeta

from ..lightning_modules.feedforward_classification import ClassificationBackbone


class _BatchNormLastDim(pt.nn.Module):
    """
    Apply BatchNorm1d to the last dimension of an arbitrary-shaped tensor.

    - For x of shape (..., F), we reshape to (-1, F), apply BN, reshape back.
    - This supports both (B, F) and (B, S, F) layouts used in DeepSet.
    """
    def __init__(
        self,
        *,
        num_features: int,
    ) -> None:
        super().__init__()
        self.bn = pt.nn.BatchNorm1d(
            num_features=num_features,
        )

    def forward(
        self,
        x: pt.Tensor,
    ) -> pt.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        x2 = self.bn(x2)
        return x2.reshape(orig_shape)


def _make_norm1d(
    *,
    norm_type: Literal["none", "batch", "layer"],
    num_features: int,
) -> pt.nn.Module:
    if norm_type == "none":
        return pt.nn.Identity()

    if norm_type == "batch":
        return _BatchNormLastDim(
            num_features=num_features,
        )

    if norm_type == "layer":
        return pt.nn.LayerNorm(
            normalized_shape=num_features,
        )

    raise ValueError(f"Unknown norm_type={norm_type!r}.")


def _make_linear_block(
    *,
    in_dim: int,
    out_dim: int,
    activation: pt.nn.Module,
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
        activation,
    ]
    if dropout is not None:
        layers.append(
            pt.nn.Dropout(p=dropout),
        )
    return pt.nn.Sequential(*layers)


class FeedforwardClassificationBackbone(
    ClassificationBackbone,
    Configurable,
    metaclass=ConfigurableABCMeta,
):
    hidden_size: int = 128
    num_hidden_layers: int = 1

    activation: pt.nn.Module = field(default_factory=pt.nn.LeakyReLU)
    use_residual_connections: bool = True
    dropout: float = 0.0

    # NEW
    ff_norm_type: Literal["none", "batch", "layer"] = "none"

    def __init__(
        self,
        *,
        input_size: int = None,
        use_deepset_encoding: bool = False,
        bootstrap_size: int = None,
        embedding_size: int = 64,
        **kwargs,
    ):
        super().__init__(
            embedding_size=embedding_size,
            **kwargs,
        )
        self.input_size = input_size
        self.use_deepset_encoding = use_deepset_encoding

        if bootstrap_size is not None:
            assert use_deepset_encoding
        self.bootstrap_size = bootstrap_size
        self.embedding_size = embedding_size

        if self.use_deepset_encoding:
            assert (self.input_size % self.bootstrap_size) == 0
            self.input_size = self.input_size // self.bootstrap_size

        dropout = self.dropout if self.dropout > 0.0 else None

        if self.use_deepset_encoding:
            if self.num_hidden_layers > 0:
                self.deepset_input_layer = _make_linear_block(
                    in_dim=self.input_size,
                    out_dim=self.hidden_size,
                    activation=self.activation,
                    dropout=dropout,
                    norm_type=self.ff_norm_type,
                )
                self.deepset_hidden_layers = pt.nn.Sequential(
                    *[
                        _make_linear_block(
                            in_dim=self.hidden_size,
                            out_dim=self.hidden_size,
                            activation=self.activation,
                            dropout=dropout,
                            norm_type=self.ff_norm_type,
                        )
                        for _ in range(self.num_hidden_layers)
                    ]
                )
                # IMPORTANT: no normalization on embedding layer
                self.deepset_embedding_layer = _make_linear_block(
                    in_dim=self.hidden_size,
                    out_dim=self.embedding_size,
                    activation=self.activation,
                    dropout=dropout,
                    norm_type="none",
                )
            else:
                self.deepset_embedding_layer = _make_linear_block(
                    in_dim=self.input_size,
                    out_dim=self.embedding_size,
                    activation=self.activation,
                    dropout=dropout,
                    norm_type="none",
                )

        ff_input_dim = self.input_size if not self.use_deepset_encoding else self.embedding_size

        if self.num_hidden_layers > 0:
            self.input_layer = _make_linear_block(
                in_dim=ff_input_dim,
                out_dim=self.hidden_size,
                activation=self.activation,
                dropout=dropout,
                norm_type=self.ff_norm_type,
            )
            self.hidden_layers = pt.nn.Sequential(
                *[
                    _make_linear_block(
                        in_dim=self.hidden_size,
                        out_dim=self.hidden_size,
                        activation=self.activation,
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
                activation=self.activation,
                dropout=dropout,
                norm_type="none",
            )
        else:
            self.embedding_layer = _make_linear_block(
                in_dim=ff_input_dim,
                out_dim=self.embedding_size,
                activation=self.activation,
                dropout=dropout,
                norm_type="none",
            )

    def forward(
        self,
        x: pt.Tensor,
        dtype: pt.dtype = None,
    ) -> pt.Tensor:
        if dtype is not None:
            x = x.to(dtype=dtype)

        if self.use_deepset_encoding:
            assert x.shape[-1] == (self.input_size * self.bootstrap_size)
            x = x.view(
                x.shape[0],
                self.bootstrap_size,
                self.input_size,
            )

            if self.num_hidden_layers > 0:
                x = self.deepset_input_layer(x)

                for hidden_layer in self.deepset_hidden_layers:
                    x_in = x
                    x = hidden_layer(x)
                    if self.use_residual_connections:
                        x = x + x_in

                x = self.deepset_embedding_layer(x)
            else:
                x = self.deepset_embedding_layer(x)

            x = x.sum(dim=1)

        if self.num_hidden_layers > 0:
            x = self.input_layer(x)

            for hidden_layer in self.hidden_layers:
                x_in = x
                x = hidden_layer(x)
                if self.use_residual_connections:
                    x = x + x_in

            x = self.embedding_layer(x)
        else:
            x = self.embedding_layer(x)

        return x

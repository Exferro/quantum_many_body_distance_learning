# file: born2disc/classification_backbones/pairwise_additive.py
import torch as pt

from nestconf import Configurable, ConfigurableABCMeta

from ..lightning_modules.feedforward_classification import ClassificationBackbone


class PairwiseAdditiveClassificationBackbone(
    ClassificationBackbone,
    Configurable,
    metaclass=ConfigurableABCMeta,
):
    """
    Multiclass strictly-unary+pairwise additive logits model:

        l_k(x) = b_k + sum_i h_{k,i}(x_i) + sum_{i<j} J_{k,ij}(x_i, x_j)

    with x_i in {-1,0,1}.

    Implementation:
      - ignore first `skip_first_n` columns (e.g. basis label)
      - map {-1,0,1} -> {0,1,2} via (x + 1)
      - all pairs i<j
      - gather-sum via EmbeddingBag (no explicit one-hots)

    Notes:
      - This backbone returns logits directly, i.e. embedding_size == num_classes.
      - Symmetry pooling must be disabled to preserve strict pairwise-additive form.
    """

    # nestconf-configurable knobs
    skip_first_n: int = 1
    include_unary: bool = True
    include_pairwise: bool = True
    num_states: int = 3

    # Interface flags used by ClassificationLightningModule
    produces_logits: bool = True
    supports_symmetry_pooling: bool = False

    # Keep these for compatibility; symmetry pooling will be disabled anyway
    basis_label_included: bool = False
    basis_label_size: int = 0
    input_channels: int = 1
    input_shape = None

    def __init__(
        self,
        *,
        input_size: int = None,
        num_classes: int = None,
        embedding_size: int = None,
        **kwargs,
    ):
        if num_classes is None:
            raise ValueError("num_classes must be provided.")
        if input_size is None:
            raise ValueError("input_size must be provided.")

        # We return logits, so embedding_size must match num_classes.
        if embedding_size is None:
            embedding_size = int(num_classes)
        if int(embedding_size) != int(num_classes):
            raise ValueError(
                f"PairwiseAdditiveClassificationBackbone returns logits, so embedding_size must equal num_classes. "
                f"Got embedding_size={embedding_size}, num_classes={num_classes}."
            )

        super().__init__(
            embedding_size=int(embedding_size),
            **kwargs,
        )

        if int(self.num_states) != 3:
            raise ValueError(f"Only num_states=3 is supported, got {self.num_states}.")

        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.num_sites = int(self.input_size - int(self.skip_first_n))

        if self.num_sites <= 0:
            raise ValueError(
                f"After skip_first_n={self.skip_first_n}, num_sites must be positive. "
                f"Got input_size={self.input_size} -> num_sites={self.num_sites}."
            )

        edges_i, edges_j = pt.triu_indices(
            row=self.num_sites,
            col=self.num_sites,
            offset=1,
        )
        self.register_buffer(
            name="edges_i",
            tensor=edges_i.to(dtype=pt.int64),
            persistent=False,
        )
        self.register_buffer(
            name="edges_j",
            tensor=edges_j.to(dtype=pt.int64),
            persistent=False,
        )

        self.num_edges = int(self.edges_i.numel())

        site_ids = pt.arange(
            self.num_sites,
            dtype=pt.int64,
        )
        edge_ids = pt.arange(
            self.num_edges,
            dtype=pt.int64,
        )
        self.register_buffer(
            name="site_ids",
            tensor=site_ids,
            persistent=False,
        )
        self.register_buffer(
            name="edge_ids",
            tensor=edge_ids,
            persistent=False,
        )

        if bool(self.include_unary):
            self.unary_bag = pt.nn.EmbeddingBag(
                num_embeddings=self.num_sites * self.num_states,
                embedding_dim=self.num_classes,
                mode="sum",
            )
            pt.nn.init.zeros_(self.unary_bag.weight)
        else:
            self.unary_bag = None

        if bool(self.include_pairwise):
            self.pair_bag = pt.nn.EmbeddingBag(
                num_embeddings=self.num_edges * (self.num_states * self.num_states),
                embedding_dim=self.num_classes,
                mode="sum",
            )
            pt.nn.init.zeros_(self.pair_bag.weight)
        else:
            self.pair_bag = None

        self.bias = pt.nn.Parameter(
            pt.zeros(self.num_classes),
            requires_grad=True,
        )

    def _to_state_indices(
        self,
        *,
        x_sites: pt.Tensor,
    ) -> pt.Tensor:
        # snapshots are stored as float dtype in your pipeline; make this robust
        x_int = pt.round(x_sites).to(dtype=pt.int64)
        x_idx = x_int + 1  # {-1,0,1} -> {0,1,2}

        if pt.any((x_idx < 0) | (x_idx >= self.num_states)):
            bad = x_sites[(x_idx < 0) | (x_idx >= self.num_states)]
            raise ValueError(
                f"Found snapshot values outside {{-1,0,1}}. Example bad values: {bad[:10].tolist()}"
            )

        return x_idx

    def forward(
        self,
        x: pt.Tensor,
        dtype: pt.dtype = None,
    ) -> pt.Tensor:
        # Drop leading metadata columns (basis label, etc.)
        x_sites = x[:, int(self.skip_first_n):]

        x_idx = self._to_state_indices(
            x_sites=x_sites,
        )  # [B, L] in {0,1,2}, int64

        bsz = int(x_idx.shape[0])
        device = x_idx.device

        logits = self.bias.unsqueeze(0).expand(bsz, -1)

        if self.unary_bag is not None:
            site_ids = self.site_ids.to(device=device)

            unary_indices = site_ids.unsqueeze(0) * self.num_states + x_idx  # [B, L]
            unary_indices_flat = unary_indices.reshape(-1)

            unary_offsets = pt.arange(
                start=0,
                end=bsz * self.num_sites,
                step=self.num_sites,
                device=device,
                dtype=pt.int64,
            )

            unary_sum = self.unary_bag(
                unary_indices_flat,
                unary_offsets,
            )  # [B, K]

            logits = logits + unary_sum

        if self.pair_bag is not None:
            edges_i = self.edges_i.to(device=device)
            edges_j = self.edges_j.to(device=device)
            edge_ids = self.edge_ids.to(device=device)

            xi = x_idx[:, edges_i]  # [B, E]
            xj = x_idx[:, edges_j]  # [B, E]

            ab = self.num_states * xi + xj  # [B, E] in {0..8}

            pair_indices = edge_ids.unsqueeze(0) * (self.num_states * self.num_states) + ab  # [B, E]
            pair_indices_flat = pair_indices.reshape(-1)

            pair_offsets = pt.arange(
                start=0,
                end=bsz * self.num_edges,
                step=self.num_edges,
                device=device,
                dtype=pt.int64,
            )

            pair_sum = self.pair_bag(
                pair_indices_flat,
                pair_offsets,
            )  # [B, K]

            logits = logits + pair_sum

        if dtype is not None:
            logits = logits.to(dtype=dtype)

        return logits

import torch as pt
from typing import List, Tuple


def aggregate_seeded_results(
    *, 
    seeded_f_tensors: List[pt.Tensor], 
    seeded_std_tensors: List[pt.Tensor], 
    use_inverse_variance_weighting: bool = False,
    stability_epsilon: float = 1e-8,
    verbose: bool = False
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Aggregates ensemble results from multiple seeds and returns the
    STANDARD ERROR of the aggregated mean, decomposed into aleatoric
    and epistemic contributions.

    Args:
        seeded_f_tensors: List of M tensors (mean estimates from each seed).
        seeded_std_tensors: List of M tensors (standard errors of those means).
        use_inverse_variance_weighting: If True, uses inverse-variance weighting
            to build the consensus mean; otherwise, uses simple unweighted mean.
        stability_epsilon: Small constant to prevent division by zero in IVW.

    Returns:
        (
            aggregated_mean,
            total_uncertainty,      # std error of aggregated_mean
            aleatoric_uncertainty,  # aleatoric std error of aggregated_mean
            epistemic_uncertainty,  # epistemic std error of aggregated_mean
        )
    """
    if verbose:
        print(f'Aggregating results from {len(seeded_f_tensors)} seeds.')

    # 1. Stack inputs to introduce batch dimension [M, *shape]
    input_means = pt.stack(seeded_f_tensors)          # [M, *shape]
    std_errors_of_means = pt.stack(seeded_std_tensors)  # [M, *shape]
    input_variances = std_errors_of_means ** 2        # [M, *shape]

    num_seeds = input_means.shape[0]

    # --- Weighting Logic ---
    if use_inverse_variance_weighting:
        # Precision weights: w_i ∝ 1 / σ_i^2
        inverse_variances = 1.0 / (input_variances + stability_epsilon)  # [M, *shape]
        normalized_weights = inverse_variances / inverse_variances.sum(
            dim=0,
            keepdim=True
        )  # [M, *shape], sum_i w_i = 1 at each location
    else:
        # Unweighted mean: all seeds contribute equally
        normalized_weights = pt.ones_like(input_means) / float(num_seeds)  # [M, *shape]

    # 2. Aggregated mean (works for both weighted and unweighted)
    aggregated_mean = (input_means * normalized_weights).sum(dim=0)  # [*shape]

    # 3. Aleatoric contribution to VARIANCE of aggregated_mean
    #
    # If seed i has variance σ_i^2 for its mean, and we take
    #    μ_agg = Σ_i w_i μ_i
    # then Var_aleatoric(μ_agg) = Σ_i w_i^2 σ_i^2
    #
    aleatoric_variance = (input_variances * (normalized_weights ** 2)).sum(
        dim=0
    )  # [*shape]

    # 4. Epistemic contribution: first estimate variance across seeds
    #
    # We treat the seed means as μ_i = θ_i + ε_i, where θ_i are "true"
    # per-seed parameters and ε_i is sampling noise.
    # Var_across_seeds(μ_i) ≈ Var(θ_i) + average(σ_i^2), but we use
    # the simple across-seed variance of μ_i as an estimator for Var(θ_i)
    # up to small corrections. This is your "epistemic variance per seed".
    #
    squared_deviation_from_consensus = (
        input_means - aggregated_mean.unsqueeze(0)
    ) ** 2  # [M, *shape]

    # Population-style variance across seeds (no Bessel correction).
    # If you prefer unbiased, use num_seeds - 1 instead of num_seeds.
    epistemic_variance_across_seeds = squared_deviation_from_consensus.mean(
        dim=0
    )  # [*shape], ~ Var_seed(μ_i)

    # Now convert this into VARIANCE of the (weighted) mean.
    #
    # If Var_seed(μ_i) = σ_theta^2 and μ_agg = Σ_i w_i μ_i with independent seeds,
    # then Var_epistemic(μ_agg) = σ_theta^2 Σ_i w_i^2.
    #
    weight_square_sum = (normalized_weights ** 2).sum(dim=0)  # [*shape]
    epistemic_variance = epistemic_variance_across_seeds * weight_square_sum  # [*shape]

    # 5. Total variance of the aggregated mean
    total_variance = aleatoric_variance + epistemic_variance  # [*shape]

    # 6. Return standard deviations (standard errors of the aggregated mean)
    return (
        aggregated_mean,
        pt.sqrt(total_variance),
        pt.sqrt(aleatoric_variance),
        pt.sqrt(epistemic_variance),
    )

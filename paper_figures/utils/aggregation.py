from typing import List, Tuple

import torch as pt


def aggregate_seeded_results(
    *,
    seeded_f_tensors: List[pt.Tensor],
    seeded_std_tensors: List[pt.Tensor],
    use_inverse_variance_weighting: bool = False,
    stability_epsilon: float = 1e-8,
    verbose: bool = False,
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    """Aggregate per-seed tensors and return mean + uncertainty decomposition."""
    if verbose:
        print(f"Aggregating results from {len(seeded_f_tensors)} seeds.")

    input_means = pt.stack(seeded_f_tensors)
    std_errors_of_means = pt.stack(seeded_std_tensors)
    input_variances = std_errors_of_means**2
    num_seeds = input_means.shape[0]

    if use_inverse_variance_weighting:
        inverse_variances = 1.0 / (input_variances + stability_epsilon)
        normalized_weights = inverse_variances / inverse_variances.sum(dim=0, keepdim=True)
    else:
        normalized_weights = pt.ones_like(input_means) / float(num_seeds)

    aggregated_mean = (input_means * normalized_weights).sum(dim=0)

    aleatoric_variance = (input_variances * (normalized_weights**2)).sum(dim=0)

    squared_deviation_from_consensus = (input_means - aggregated_mean.unsqueeze(0)) ** 2
    epistemic_variance_across_seeds = squared_deviation_from_consensus.mean(dim=0)
    weight_square_sum = (normalized_weights**2).sum(dim=0)
    epistemic_variance = epistemic_variance_across_seeds * weight_square_sum

    total_variance = aleatoric_variance + epistemic_variance

    return (
        aggregated_mean,
        pt.sqrt(total_variance),
        pt.sqrt(aleatoric_variance),
        pt.sqrt(epistemic_variance),
    )


"""Independent loss weighting for distributional IQA pair training."""


def pair_score_objective(
    rank_loss,
    next_token_loss,
    softkl_loss,
    in_level_loss,
    *,
    weight_rank=1.0,
    weight_next_token=0.05,
    weight_softkl=1.0,
    weight_in_level=None,
):
    weights = (weight_rank, weight_next_token, weight_softkl,
               0.0 if weight_in_level is None else weight_in_level)
    if any(weight < 0 for weight in weights):
        raise ValueError("Loss weights must be nonnegative")
    return (weights[0] * rank_loss + weights[1] * next_token_loss
            + weights[2] * softkl_loss + weights[3] * in_level_loss)

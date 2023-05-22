from typing import Optional
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state


def obtain_candidate_reward_structures(
    len_list: int,
    random_state: Optional[int] = None,
):
    random_ = check_random_state(random_state)

    standard = np.ones((len_list, len_list))
    cascade = np.tril(np.ones((len_list, len_list)))
    independent = np.eye(len_list)
    all_pos = np.arange(len_list)

    cascade_neighbor_1 = cascade.copy()
    independent_neighbor_1 = independent.copy()
    for pos_ in range(len_list - 1):
        cascade_neighbor_1[pos_, pos_ + 1] = True
        cascade_neighbor_1[pos_ + 1, pos_] = True
        independent_neighbor_1[pos_, pos_ + 1] = True
        independent_neighbor_1[pos_ + 1, pos_] = True

    cascade_neighbor_2 = cascade_neighbor_1.copy()
    independent_neighbor_2 = independent_neighbor_1.copy()
    for pos_ in range(len_list - 2):
        cascade_neighbor_2[pos_, pos_ + 2] = True
        cascade_neighbor_2[pos_ + 2, pos_] = True
        independent_neighbor_2[pos_, pos_ + 2] = True
        independent_neighbor_2[pos_ + 2, pos_] = True

    cascade_neighbor_3 = cascade_neighbor_2.copy()
    independent_neighbor_3 = independent_neighbor_2.copy()
    for pos_ in range(len_list - 3):
        cascade_neighbor_3[pos_, pos_ + 3] = True
        cascade_neighbor_3[pos_ + 3, pos_] = True
        independent_neighbor_3[pos_, pos_ + 3] = True
        independent_neighbor_3[pos_ + 3, pos_] = True

    random_6 = independent.copy()
    random_3 = independent.copy()
    for pos_ in range(len_list):
        candidate_pos = np.setdiff1d(all_pos, np.array([pos_]))
        random_pos_6 = random_.choice(candidate_pos, size=6)
        random_pos_3 = random_.choice(candidate_pos, size=3)
        random_6[pos_, random_pos_6] = True
        random_3[pos_, random_pos_3] = True

    candidate_reward_structures = [
        standard,
        random_6,
        random_3,
        cascade_neighbor_3,
        cascade_neighbor_2,
        independent_neighbor_1,
    ]
    candidate_reward_structure_num_random_actions = np.array([-1, -1, -1, -1, -1, -1])
    return candidate_reward_structures, candidate_reward_structure_num_random_actions


def obtain_candidate_reward_structure_coefs(
    dim_context: int,
    n_candidate_reward_structures: int,
    random_state: Optional[int] = None,
):
    random_ = check_random_state(random_state)
    candidate_reward_structure_coefs = random_.uniform(
        size=(n_candidate_reward_structures, dim_context)
    )
    # align coef based on similarity (similar reward structure has similar coef)
    distance = np.zeros((n_candidate_reward_structures, n_candidate_reward_structures))
    for structure_1 in range(n_candidate_reward_structures - 1):
        for structure_2 in range(structure_1 + 1, n_candidate_reward_structures):
            distance[structure_1, structure_2] = distance[
                structure_2, structure_1
            ] = mean_squared_error(
                candidate_reward_structure_coefs[structure_1],
                candidate_reward_structure_coefs[structure_2],
            )
    coef_order = np.zeros(n_candidate_reward_structures, dtype=int)
    coef_order[0] = np.argmax(distance.sum(axis=1))
    distance[:, coef_order[0]] = np.full((n_candidate_reward_structures,), np.infty)
    distance_ = distance[coef_order[0]]
    for i in range(1, n_candidate_reward_structures):
        coef_order[i] = np.argmin(distance_)
        distance[:, coef_order[i]] = np.full((n_candidate_reward_structures,), np.infty)
        distance_ = distance_ + distance[coef_order[i]]
    return candidate_reward_structure_coefs[coef_order]


def diverse_reward_structure(
    len_list: int,
    dim_context: int,
    weight_scaler: Optional[float] = None,
    weight_interpolation_param: Optional[float] = None,
    random_state: Optional[int] = None,
):
    if weight_scaler is None and weight_interpolation_param is None:
        raise ValueError(
            "either weight_scaler or weight_interpolation_param must be given"
        )
    elif weight_scaler is not None and weight_interpolation_param is not None:
        warnings.warn(
            "both weight_scaler and weight_interpolation_param is given. weight_interpolation_param will be prioritized."
        )

    (
        candidate_reward_structures,
        candidate_reward_structures_num_random_actions,
    ) = obtain_candidate_reward_structures(len_list=len_list, random_state=random_state)
    n_candidate_reward_structures = len(candidate_reward_structures)

    candidate_reward_structure_coefs = obtain_candidate_reward_structure_coefs(
        dim_context=dim_context,
        n_candidate_reward_structures=n_candidate_reward_structures,
        random_state=random_state,
    )

    if weight_scaler is not None:
        candidate_reward_structure_weights = np.full(
            (n_candidate_reward_structures,), weight_scaler
        )
    if weight_interpolation_param is not None:
        weight_coef = np.arange(n_candidate_reward_structures)
        weight_coef = weight_coef - weight_coef.mean()
        weight_coef = -weight_coef * 1.5 / weight_coef.max()

        candidate_reward_structure_weights = np.exp(
            weight_interpolation_param * weight_coef
        ) * np.exp((1 - weight_interpolation_param) * (-weight_coef))

    return (
        candidate_reward_structures,
        candidate_reward_structures_num_random_actions,
        candidate_reward_structure_weights,
        candidate_reward_structure_coefs,
    )

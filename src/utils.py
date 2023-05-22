"""Useful Tools."""
from typing import Optional

import numpy as np
import pandas as pd

from obp.utils import _check_slate_ope_inputs


def check_aips_inputs(
    slate_id: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    pscore_given_user_behavior_model: np.ndarray,
    evaluation_policy_pscore_given_user_behavior_model: np.ndarray,
) -> Optional[ValueError]:
    """Check inputs of SlateAdaptiveIPS.

    Parameters
    -----------
    slate_id: array-like, shape (<= n_rounds * len_list,)
        Slate id observed for each data in logged bandit data.

    reward: array-like, shape (<= n_rounds * len_list,)
        Slot-level rewards, i.e., :math:`r_{i}(l)`.

    position: array-like, shape (<= n_rounds * len_list,)
        Indices to differentiate positions in a recommendation interface where the actions are presented.

    pscore_given_user_behavior_model: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

    evaluation_policy_pscore_given_user_behavior_model: array-like, shape (<= n_rounds * len_list,)
        Action choice probabilities of the evaluation policy, i.e., :math:`\\pi_e(a_i|x_i)`.

    """
    _check_slate_ope_inputs(
        slate_id=slate_id,
        reward=reward,
        position=position,
        pscore=pscore_given_user_behavior_model,
        evaluation_policy_pscore=evaluation_policy_pscore_given_user_behavior_model,
        pscore_type="pscore_given_user_behavior_model",
    )

    bandit_feedback_df = pd.DataFrame()
    bandit_feedback_df["slate_id"] = slate_id
    bandit_feedback_df["position"] = position
    # check uniqueness
    if bandit_feedback_df.duplicated(["slate_id", "position"]).sum() > 0:
        raise ValueError("`position` must not be duplicated in each slate")


def format_runtime(start: int, finish: int):
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"

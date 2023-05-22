"""Off-Policy Estimators for Slate/Ranking Policies."""
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np

from obp.ope.estimators_slate import BaseSlateInverseProbabilityWeighting

from utils import check_aips_inputs


@dataclass
class SlateAdaptiveIPS(BaseSlateInverseProbabilityWeighting):
    """Adaptive Inverse Propensity Scoring (AIPS) Estimator.

    Note
    -------
    AIPS estimates the policy value of evaluation policy :math:`\\pi_e`
    with the provided click model.

    Parameters
    ----------
    estimator_name: str, default='aips'.
        Name of the estimator.

    References
    ------------

    """

    estimator_name: str = "aips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_given_user_behavior_model: np.ndarray,
        evaluation_policy_pscore_given_user_behavior_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_given_user_behavior_model: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore_given_user_behavior_model: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.
        """
        check_aips_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_given_user_behavior_model=pscore_given_user_behavior_model,
            evaluation_policy_pscore_given_user_behavior_model=evaluation_policy_pscore_given_user_behavior_model,
        )
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore_given_user_behavior_model,
                evaluation_policy_pscore=evaluation_policy_pscore_given_user_behavior_model,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_given_user_behavior_model: np.ndarray,
        evaluation_policy_pscore_given_user_behavior_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_given_user_behavior_model: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore_given_user_behavior_model: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_aips_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_given_user_behavior_model=pscore_given_user_behavior_model,
            evaluation_policy_pscore_given_user_behavior_model=evaluation_policy_pscore_given_user_behavior_model,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_given_user_behavior_model,
            evaluation_policy_pscore=evaluation_policy_pscore_given_user_behavior_model,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

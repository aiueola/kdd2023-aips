from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Callable, Optional, List

import numpy as np
from sklearn.utils import check_random_state

from obp.types import BanditFeedback

from synthetic import SyntheticSlateBanditDataset


@dataclass
class Node:
    node_id: int
    user_behavior: int
    depth: int
    sample_id: Optional[np.ndarray] = None
    bootstrap_sample_ids: Optional[List[np.ndarray]] = None


@dataclass
class UserBehaviorTree:
    dataset: SyntheticSlateBanditDataset
    candidate_models: List[np.ndarray]
    behavior_policy_function: Callable
    evaluation_policy_beta: float
    evaluation_policy_epsilon: float
    len_list: int
    n_partition: int = 10
    min_samples_leaf: int = 10
    max_depth: Optional[int] = None
    noise_level: float = 0.0
    n_bootstrap: int = 10
    random_state: Optional[int] = None

    def __post_init__(self):
        self.decision_boundary = [dict() for _ in range(self.len_list)]
        self.position_wise_expected_reward = np.zeros(self.len_list)
        val_dataset = self.dataset.obtain_batch_bandit_feedback(n_rounds=10000)
        behavior_policy_logit_ = self.behavior_policy_function(
            context=val_dataset["context"],
            action_context=self.dataset.action_context,
            random_state=self.dataset.random_state,
        )
        evaluation_policy_logit_ = self.evaluation_policy_beta * behavior_policy_logit_
        self.position_wise_expected_reward = (
            self.dataset.calc_position_wise_on_policy_policy_value(
                context=val_dataset["context"],
                evaluation_policy_logit_=evaluation_policy_logit_,
                epsilon=self.evaluation_policy_epsilon,
            )
        )
        # key: node_id, value: (parent_user_behavior, split_exist, feature_dim, feature_value)
        if self.max_depth is None:
            self.max_depth = np.infty
        self.random_ = check_random_state(self.random_state)

    def cross_fitting(
        self,
        position: int,
        context: np.ndarray,  # (n_samples, n_features)
        behavior_policy_pscore: np.ndarray,  # (n_candidate_models, n_samples * len_list)
        evaluation_policy_pscore: np.ndarray,  # (n_candidate_models, n_samples * len_list)
        reward: np.ndarray,  # (n_samples, )
        n_partition: int = 3,
    ):
        n_candidate_models, n_samples = behavior_policy_pscore.shape
        n_samples = n_samples // self.len_list
        partition = self.random_.choice(
            n_candidate_models, size=n_samples, replace=True
        )
        behavior_policy_pscore = behavior_policy_pscore.reshape(
            (n_candidate_models, n_samples, self.len_list)
        )
        evaluation_policy_pscore = evaluation_policy_pscore.reshape(
            (n_candidate_models, n_samples, self.len_list)
        )
        reward = reward.reshape((n_samples, self.len_list))

        user_behavior_model = np.zeros(n_samples)
        for partition_id in range(n_partition):
            train_id_ = partition != partition_id
            self.fit(
                context=context[train_id_],
                behavior_policy_pscore=behavior_policy_pscore[:, train_id_, :],
                evaluation_policy_pscore=evaluation_policy_pscore[:, train_id_, :],
                reward=reward[train_id_, :],
                position=position,
                n_total_samples=n_samples,
            )

            test_id_ = partition == partition_id
            user_behavior_model[np.arange(n_samples)[test_id_]] = self.predict(
                context=context[test_id_, :],
                position=position,
            )

        return user_behavior_model

    def fit_predict(
        self,
        context: np.ndarray,  # (n_samples, n_features)
        behavior_policy_pscore: np.ndarray,  # (n_candidate_models, n_samples * len_list)
        evaluation_policy_pscore: np.ndarray,  # (n_candidate_models, n_samples * len_list)
        reward: np.ndarray,  # (n_samples * len_list, )
        position: int,
        bootstrap_dataset: Optional[List[BanditFeedback]] = None,
    ):
        self.fit(
            context=context,
            behavior_policy_pscore=behavior_policy_pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
            reward=reward,
            position=position,
            bootstrap_dataset=bootstrap_dataset,
        )
        return self.train_reward_structure

    def fit(
        self,
        context: np.ndarray,  # (n_samples, n_features)
        behavior_policy_pscore: np.ndarray,  # (n_candidate_models, n_samples * len_list) or (n_candidate_models, n_samples, len_list)
        evaluation_policy_pscore: np.ndarray,  # (n_candidate_models, n_samples * len_list) or (n_candidate_models, n_samples, len_list)
        reward: np.ndarray,  # (n_samples * len_list, ) or (n_samples, len_list)
        position: int,
        bootstrap_dataset: Optional[List[BanditFeedback]] = None,
    ):
        self.train_n_samples, self.n_features = context.shape
        self.n_candidate_models = behavior_policy_pscore.shape[0]
        # bootstrapped dataset to calculate mse
        self.bootstrap_dataset = []
        for i in range(self.n_bootstrap):
            if bootstrap_dataset is not None:
                logged_dataset_ = bootstrap_dataset[i]
            else:
                logged_dataset_ = self.dataset.obtain_batch_bandit_feedback(
                    self.train_n_samples
                )
            context_ = logged_dataset_["context"]
            action_ = logged_dataset_["action"].reshape((-1, self.len_list))
            reward_ = logged_dataset_["reward"].reshape((-1, self.len_list))[
                :, position
            ]
            behavior_policy_logit_ = self.behavior_policy_function(
                context=context_,
                action_context=self.dataset.action_context,
                random_state=self.dataset.random_state,
            )
            evaluation_policy_logit_ = (
                self.evaluation_policy_beta * behavior_policy_logit_
            )

            behavior_policy_pscore_all_ = np.zeros(
                (self.n_candidate_models, self.train_n_samples)
            )
            evaluation_policy_pscore_all_ = np.zeros(
                (self.n_candidate_models, self.train_n_samples)
            )
            for user_behavior in range(self.n_candidate_models):
                behavior_policy_pscore_all_[
                    user_behavior
                ] = self.dataset.calc_pscore_given_reward_structure_and_position(
                    position=position,
                    policy_logit_=behavior_policy_logit_,
                    action=action_.flatten(),
                    reward_structure=np.tile(
                        self.candidate_models[user_behavior],
                        (self.train_n_samples, 1, 1),
                    ),
                )
                evaluation_policy_pscore_all_[
                    user_behavior
                ] = self.dataset.calc_pscore_given_reward_structure_and_position(
                    position=position,
                    policy_logit_=evaluation_policy_logit_,
                    action=action_.flatten(),
                    reward_structure=np.tile(
                        self.candidate_models[user_behavior],
                        (self.train_n_samples, 1, 1),
                    ),
                    epsilon=self.evaluation_policy_epsilon,
                )
            importance_weight_ = (
                evaluation_policy_pscore_all_ / behavior_policy_pscore_all_
            )

            self.bootstrap_dataset.append(
                dict(
                    context=context_,
                    reward=reward_,
                    importance_weight=importance_weight_,
                    reward_structure=np.zeros(self.train_n_samples, dtype=int),
                    weighted_reward=np.zeros(self.train_n_samples),
                )
            )
        # for ope
        behavior_policy_pscore = behavior_policy_pscore.reshape(
            (self.n_candidate_models, self.train_n_samples, self.len_list)
        )[:, :, position]
        evaluation_policy_pscore = evaluation_policy_pscore.reshape(
            (self.n_candidate_models, self.train_n_samples, self.len_list)
        )[:, :, position]
        reward = reward.reshape((self.train_n_samples, self.len_list))[:, position]

        self.train_context = context
        self.train_importance_weight = evaluation_policy_pscore / behavior_policy_pscore
        self.train_reward = reward
        self.train_reward_structure = np.zeros(self.train_n_samples, dtype=int)
        self.train_weighted_reward = np.zeros(self.train_n_samples)

        base_user_behavior = self._select_base_user_behavior(position)
        node_queue = deque()

        # initial node
        initial_node = Node(
            node_id=0,
            sample_id=np.arange(self.train_n_samples),
            user_behavior=base_user_behavior,
            depth=0,
            bootstrap_sample_ids=[
                np.arange(self.train_n_samples) for _ in range(self.n_bootstrap)
            ],
        )
        node_queue.append(initial_node)
        node_id = 0

        # update later if the child nodes exist
        self.decision_boundary[position][0] = {
            "parent_user_behavior": base_user_behavior,
            "split_exist": False,
            "feature_dim": None,
            "feature_value": None,
        }

        self._update_global_pscore(initial_node)

        while len(node_queue):
            parent_node = node_queue.pop()
            split_exist, split_outcome = self._search_split(
                parent_node=parent_node, position=position
            )

            if split_exist:
                (
                    left_sample_id,
                    right_sample_id,
                    left_user_behavior,
                    right_user_behavior,
                    left_sample_ids,
                    right_sample_ids,
                ) = split_outcome

                left_node = Node(
                    node_id=node_id + 1,
                    sample_id=left_sample_id,
                    user_behavior=left_user_behavior,
                    depth=parent_node.depth + 1,
                    bootstrap_sample_ids=left_sample_ids,
                )
                right_node = Node(
                    node_id=node_id + 2,
                    sample_id=right_sample_id,
                    user_behavior=right_user_behavior,
                    depth=parent_node.depth + 1,
                    bootstrap_sample_ids=right_sample_ids,
                )

                # update later if the child nodes exist
                self.decision_boundary[position][node_id + 1] = {
                    "parent_user_behavior": left_user_behavior,
                    "split_exist": False,
                    "feature_dim": None,
                    "feature_value": None,
                }
                self.decision_boundary[position][node_id + 2] = {
                    "parent_user_behavior": right_user_behavior,
                    "split_exist": False,
                    "feature_dim": None,
                    "feature_value": None,
                }

                self._update_global_pscore(left_node)
                self._update_global_pscore(right_node)

                node_queue.append(left_node)
                node_queue.append(right_node)
                node_id += 2

    def predict(
        self,
        context: np.ndarray,  # (n_samples, n_features)
        position: int,
    ):
        self.test_n_samples = context.shape[0]
        self.test_context = context

        node_queue = deque()
        user_behavior = np.zeros(self.test_n_samples, dtype=int)

        initial_node = Node(
            node_id=0,
            sample_id=np.arange(self.test_n_samples),
            user_behavior=self.decision_boundary[position][0]["parent_user_behavior"],
            depth=0,
        )

        node_queue.append(initial_node)
        node_id = 0

        while len(node_queue):
            parent_node = node_queue.pop()
            parent_node_id = parent_node.node_id
            parent_sample_id = parent_node.sample_id
            parent_depth = parent_node.depth
            parent_context = self.test_context[parent_sample_id]

            decision_boundary = self.decision_boundary[position][parent_node_id]
            split_exist = decision_boundary["split_exist"]

            if split_exist:
                boundary_feature = decision_boundary["feature_dim"]
                boundary_value = decision_boundary["feature_value"]

                left_sample_id = parent_sample_id[
                    np.where(parent_context[:, boundary_feature] < boundary_value)
                ]
                right_sample_id = parent_sample_id[
                    np.where(parent_context[:, boundary_feature] >= boundary_value)
                ]
                left_node = Node(
                    node_id=node_id + 1,
                    sample_id=left_sample_id,
                    user_behavior=self.decision_boundary[position][node_id + 1][
                        "parent_user_behavior"
                    ],
                    depth=parent_depth + 1,
                )
                right_node = Node(
                    node_id=node_id + 2,
                    sample_id=right_sample_id,
                    user_behavior=self.decision_boundary[position][node_id + 2][
                        "parent_user_behavior"
                    ],
                    depth=parent_depth + 1,
                )
                node_queue.append(left_node)
                node_queue.append(right_node)
                node_id += 2
            else:
                user_behavior[parent_node.sample_id] = parent_node.user_behavior

        return user_behavior

    def _select_base_user_behavior(
        self,
        position: int,
    ):
        estimate = np.zeros((self.n_candidate_models, self.n_bootstrap))
        for i in range(self.n_bootstrap):
            logged_dataset = self.bootstrap_dataset[i]
            reward = logged_dataset["reward"]
            importance_weight = logged_dataset["importance_weight"]

            for user_behavior in range(self.n_candidate_models):
                iw = importance_weight[user_behavior]
                estimate[user_behavior, i] = (iw * reward).mean()

        bias = estimate.mean(axis=1) - self.position_wise_expected_reward[position]
        bias = self.random_.normal(loc=bias, scale=np.abs(bias) * self.noise_level)

        variance = np.zeros(self.n_candidate_models)
        for user_behavior in range(self.n_candidate_models):
            estimate = self.train_importance_weight[user_behavior] * self.train_reward
            variance[user_behavior] = estimate.var(ddof=1) / self.train_n_samples

        best_user_behavior = (bias**2 + variance).argmin()
        self.minimum_variance = variance[-1]
        return best_user_behavior

    def _search_split(
        self,
        parent_node: Node,
        position: int,
    ):
        parent_node_id = parent_node.node_id
        parent_sample_id = parent_node.sample_id
        parent_bootstrap_sample_ids = parent_node.bootstrap_sample_ids
        parent_user_behavior = parent_node.user_behavior
        parent_depth = parent_node.depth
        parent_context = self.train_context[parent_sample_id]

        n_parent_samples = len(parent_sample_id)
        if n_parent_samples < 2 * self.min_samples_leaf:
            return False, None
        if parent_depth == self.max_depth:
            return False, None
        min_left_proportion = self.min_samples_leaf / n_parent_samples
        max_left_proportion = 1 - min_left_proportion

        best_mse = self._calc_mse_global(position)
        best_split_feature_dim = None
        best_split_feature_value = None
        best_left_sample_id = None
        best_right_sample_id = None
        best_left_bootstrap_sample_ids = None
        best_right_bootstrap_sample_ids = None
        best_left_user_behavior = None
        best_right_user_behavior = None

        split_feature_dims = self.random_.choice(
            self.train_context.shape[1], size=self.n_partition, replace=True
        )
        split_left_proportions = self.random_.uniform(
            min_left_proportion, max_left_proportion, size=self.n_partition
        )
        feature_dim_sort_idx = np.argsort(split_feature_dims)

        split_feature_dims = split_feature_dims[feature_dim_sort_idx]
        split_left_proportions = split_left_proportions[feature_dim_sort_idx]

        for i in range(self.n_partition):
            feature_dim = split_feature_dims[i]
            if i == 0 or split_feature_dims[i] != split_feature_dims[i - 1]:
                sorted_sample_id = parent_sample_id[
                    np.argsort(parent_context[:, feature_dim])
                ]

            split_id = int(split_left_proportions[i] * n_parent_samples)
            left_sample_id = sorted_sample_id[:split_id]
            right_sample_id = sorted_sample_id[split_id:]

            feature_value = (
                self.train_context[left_sample_id[-1], feature_dim]
                + self.train_context[right_sample_id[0], feature_dim]
            ) / 2
            (
                left_user_behavior,
                right_user_behavior,
                left_bootstrap_sample_ids,
                right_bootstrap_sample_ids,
                split_mse,
            ) = self._find_best_user_behavior(
                position=position,
                left_sample_id=left_sample_id,
                right_sample_id=right_sample_id,
                parent_sample_ids=parent_bootstrap_sample_ids,
                split_feature_dim=feature_dim,
                split_feature_value=feature_value,
            )
            if split_mse <= best_mse:
                best_mse = split_mse
                best_split_feature_dim = feature_dim
                best_split_feature_value = feature_value
                best_left_sample_id = left_sample_id
                best_right_sample_id = right_sample_id
                best_left_bootstrap_sample_ids = left_bootstrap_sample_ids
                best_right_bootstrap_sample_ids = right_bootstrap_sample_ids
                best_left_user_behavior = left_user_behavior
                best_right_user_behavior = right_user_behavior
                best_mse = split_mse

        if best_split_feature_dim is not None:  # split_exist
            split_outcome = (
                best_left_sample_id,
                best_right_sample_id,
                best_left_user_behavior,
                best_right_user_behavior,
                best_left_bootstrap_sample_ids,
                best_right_bootstrap_sample_ids,
            )
            self.decision_boundary[position][parent_node_id] = {
                "parent_user_behavior": parent_user_behavior,
                "split_exist": True,
                "feature_dim": best_split_feature_dim,
                "feature_value": best_split_feature_value,
            }
            return True, split_outcome
        else:
            return False, None

    def _find_best_user_behavior(
        self,
        position: int,
        left_sample_id: np.ndarray,
        right_sample_id: np.ndarray,
        parent_sample_ids: List[np.ndarray],
        split_feature_dim: int,
        split_feature_value: float,
    ):
        estimate = np.zeros(
            (self.n_candidate_models, self.n_candidate_models, self.n_bootstrap)
        )
        left_sample_ids = []
        right_sample_ids = []
        for i in range(self.n_bootstrap):
            sample_id = np.arange(self.train_n_samples, dtype=int)
            logged_dataset = self.bootstrap_dataset[i]
            context = logged_dataset["context"]
            reward = logged_dataset["reward"]
            importance_weight = logged_dataset["importance_weight"]
            weighted_reward_ = logged_dataset["weighted_reward"].copy()

            sample_id = parent_sample_ids[i].copy()
            context = context[sample_id]

            left_context_id = np.where(
                context[:, split_feature_dim] < split_feature_value
            )
            right_context_id = np.where(
                context[:, split_feature_dim] >= split_feature_value
            )
            left_sample_id_ = sample_id[left_context_id]
            right_sample_id_ = sample_id[right_context_id]
            left_sample_ids.append(left_sample_id_)
            right_sample_ids.append(right_sample_id_)

            for left_user_behavior_ in range(self.n_candidate_models):
                weighted_reward_[left_sample_id_] = (
                    importance_weight[left_user_behavior_, left_sample_id_]
                    * reward[left_sample_id_]
                )
                for right_user_behavior_ in range(self.n_candidate_models):
                    weighted_reward_[right_sample_id_] = (
                        importance_weight[right_user_behavior_, right_sample_id_]
                        * reward[right_sample_id_]
                    )
                    estimate[
                        left_user_behavior_, right_user_behavior_, i
                    ] = weighted_reward_.mean()

        bias = estimate.mean(axis=2) - self.position_wise_expected_reward[position]
        bias = self.random_.normal(loc=bias, scale=np.abs(bias) * self.noise_level)

        variance = np.zeros((self.n_candidate_models, self.n_candidate_models))
        weighted_reward_ = self.train_weighted_reward.copy()
        for left_user_behavior in range(self.n_candidate_models):
            weighted_reward_[left_sample_id] = (
                self.train_importance_weight[left_user_behavior, left_sample_id]
                * self.train_reward[left_sample_id]
            )
            for right_user_behavior in range(self.n_candidate_models):
                weighted_reward_[right_sample_id] = (
                    self.train_importance_weight[right_user_behavior, right_sample_id]
                    * self.train_reward[right_sample_id]
                )
                variance[left_user_behavior, right_user_behavior] = (
                    weighted_reward_.var(ddof=1) / self.train_n_samples
                )

        variance = np.clip(variance, self.minimum_variance, None)
        # print(variance)
        # print(bias ** 2)
        # print()

        mse = bias**2 + variance
        best_left_user_behavior, best_right_user_behavior = np.unravel_index(
            mse.argmin(), bias.shape
        )

        split_outcome = (
            best_left_user_behavior,
            best_right_user_behavior,
            left_sample_ids,
            right_sample_ids,
            mse.min(),
        )
        return split_outcome

    def _calc_mse_global(
        self,
        position: int,
    ):
        estimate = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            logged_dataset = self.bootstrap_dataset[i]
            estimate[i] = logged_dataset["weighted_reward"].mean()

        bias = estimate.mean() - self.position_wise_expected_reward[position]
        bias = self.random_.normal(loc=bias, scale=np.abs(bias) * self.noise_level)
        variance = self.train_weighted_reward.var(ddof=1) / self.train_n_samples
        variance = np.clip(variance, self.minimum_variance, None)
        return bias**2 + variance

    def _update_global_pscore(self, node: Node):
        user_behavior = node.user_behavior
        sample_id = node.sample_id
        bootstrap_sample_ids = node.bootstrap_sample_ids

        self.train_reward_structure[sample_id] = user_behavior
        self.train_weighted_reward[sample_id] = (
            self.train_importance_weight[user_behavior, sample_id]
            * self.train_reward[sample_id]
        )

        for i in range(self.n_bootstrap):
            sample_id = bootstrap_sample_ids[i]
            logged_dataset = self.bootstrap_dataset[i]
            iw = logged_dataset["importance_weight"]
            reward = logged_dataset["reward"]

            self.bootstrap_dataset[i]["reward_structure"][sample_id] = user_behavior
            self.bootstrap_dataset[i]["weighted_reward"][sample_id] = (
                iw[user_behavior, sample_id] * reward[sample_id]
            )

    def _record_decision_boundary(
        self,
        position: int,
        parent_node_id: int,
        parent_user_behavior: int,
        split_exist: bool,
        feature_dim: int,
        feature_value: int,
    ):
        self.decision_boundary[position][parent_node_id] = {
            "parent_user_behavior": parent_user_behavior,
            "split_exist": split_exist,
            "feature_dim": feature_dim,
            "feature_value": feature_value,
        }

    def print_tree(self, position: int):
        logs = defaultdict(list)

        node_queue = deque()
        initial_user_behavior = self.decision_boundary[position][0][
            "parent_user_behavior"
        ]
        initial_node = Node(node_id=0, user_behavior=initial_user_behavior, depth=0)
        node_queue.append(initial_node)
        node_id = 0

        while len(node_queue):
            parent_node = node_queue.pop()
            parent_node_id = parent_node.node_id
            parent_user_behavior = parent_node.user_behavior
            parent_depth = parent_node.depth

            logs[parent_depth].append(parent_user_behavior)

            decision_boundary = self.decision_boundary[position][parent_node_id]
            split_exist = decision_boundary["split_exist"]

            if split_exist:
                left_node = Node(
                    node_id=node_id + 1,
                    user_behavior=self.decision_boundary[position][node_id + 1][
                        "parent_user_behavior"
                    ],
                    depth=parent_depth + 1,
                )
                right_node = Node(
                    node_id=node_id + 2,
                    user_behavior=self.decision_boundary[position][node_id + 2][
                        "parent_user_behavior"
                    ],
                    depth=parent_depth + 1,
                )
                node_queue.append(left_node)
                node_queue.append(right_node)
                node_id += 2

        for depth in range(self.max_depth):
            print(logs[depth])
        print()

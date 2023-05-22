from typing import Dict, Any, Union, Optional, List
from pathlib import Path
from tqdm import tqdm
import copy
import time
import pickle
import hydra
import os
from omegaconf import DictConfig, ListConfig
from multiprocessing import Pool, cpu_count

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


from obp.ope import (
    SlateStandardIPS,
    SlateIndependentIPS,
    SlateRewardInteractionIPS,
)
from obp.dataset import (
    logistic_reward_function,
    linear_behavior_policy_logit,
)

from synthetic import (
    SyntheticSlateBanditDataset,
    linear_user_behavior_model,
    distance_user_behavior_model,
)
from meta_slate import SlateOffPolicyEvaluation
from estimators_slate import SlateAdaptiveIPS
from user_behavior_model import (
    diverse_reward_structure,
    obtain_candidate_reward_structures,
)
from utils import format_runtime


def generate_and_obtain_dataset(
    estimator_name: str,
    n_rounds: int,
    n_unique_action: int,
    len_list: int,
    dim_context: int,
    reward_type: str,
    reward_structure_type: str,
    decay_function: str,
    behavior_policy: str,  # "linear or "uniform"
    evaluation_policy_beta: float,
    epsilon: Optional[float],
    user_behavior_function: str,  # "linear" or "distance"
    user_behavior_weight_scaler: Union[str, int],
    user_behavior_interpolation_param: Union[str, int],
    candidate_reward_structures: List[np.ndarray],
    candidate_reward_structure_num_random_actions: np.ndarray,
    candidate_reward_structure_weights: np.ndarray,
    candidate_reward_structure_coefs: np.ndarray,
    is_deterministic_user_behavior: bool,
    is_factorizable: bool,
    reward_std: float,
    reward_scale: float,
    interaction_scale: float,
    normalize_interaction: bool,
    evaluation: str,  # "ground_truth" or "on_policy"
    random_state: int,
    base_random_state: int,
    log_dir: str,
):
    decay_function_ = decay_function if reward_structure_type == "decay" else "none"
    is_factorizable_ = "_factorizable" if is_factorizable else ""
    is_deterministic_user_behavior_ = (
        "_determinictic" if is_deterministic_user_behavior else ""
    )
    path_ = Path(
        log_dir
        + f"/dataset/{estimator_name}/{reward_type}_{reward_structure_type}_{decay_function_}"
    )
    path_.mkdir(exist_ok=True, parents=True)

    path_dataset = Path(
        path_
        / f"dataset_{behavior_policy}{is_factorizable_}_{n_unique_action}_{len_list}_{dim_context}_{user_behavior_weight_scaler}_{user_behavior_interpolation_param}_{user_behavior_function}{is_deterministic_user_behavior_}_{reward_std}_{reward_scale}_{interaction_scale}{normalize_interaction}_{base_random_state}.pickle"
    )
    path_bandit_feedback = Path(
        path_
        / f"bandit_feedback_{behavior_policy}{is_factorizable_}_{evaluation_policy_beta}_{epsilon}_{n_rounds}_{n_unique_action}_{len_list}_{dim_context}_{user_behavior_weight_scaler}_{user_behavior_interpolation_param}_{user_behavior_function}{is_deterministic_user_behavior_}_{reward_std}_{reward_scale}_{interaction_scale}{normalize_interaction}_{evaluation}_{base_random_state}_{random_state}.pickle"
    )
    behavior_policy_function = (
        linear_behavior_policy_logit if behavior_policy == "linear" else None
    )
    user_behavior_function = (
        linear_user_behavior_model
        if user_behavior_function == "linear"
        else distance_user_behavior_model
    )

    if path_bandit_feedback.exists():
        with open(path_dataset, "rb") as f:
            dataset = pickle.load(f)
        with open(path_bandit_feedback, "rb") as f:
            bandit_feedback = pickle.load(f)
        return dataset, bandit_feedback, behavior_policy_function

    if path_dataset.exists():
        with open(path_dataset, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = SyntheticSlateBanditDataset(
            n_unique_action=n_unique_action,
            len_list=len_list,
            dim_context=dim_context,
            reward_type=reward_type,
            reward_structure_type=reward_structure_type,
            decay_function=decay_function,
            behavior_policy_function=behavior_policy_function,
            is_factorizable=is_factorizable,
            candidate_reward_structures=candidate_reward_structures,
            candidate_reward_structure_num_random_actions=candidate_reward_structure_num_random_actions,
            candidate_reward_structure_weights=candidate_reward_structure_weights,
            candidate_reward_structure_coefs=candidate_reward_structure_coefs,
            user_behavior_model=user_behavior_function,
            is_deterministic_user_behavior=is_deterministic_user_behavior,
            base_reward_function=logistic_reward_function,
            reward_std=reward_std,
            reward_scale=reward_scale,
            interaction_scale=interaction_scale,
            normalize_interaction=normalize_interaction,
            random_state=base_random_state,
        )
        with open(path_dataset, "wb") as f:
            pickle.dump(dataset, f)

    # set random state
    dataset.random_ = check_random_state(random_state)
    # script
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds,
        return_pscore=(estimator_name == "IPS"),
        return_pscore_cascade=(estimator_name == "RIPS"),
        return_pscore_item_position=(estimator_name == "IIPS"),
    )

    if behavior_policy_function is None:  # uniform random
        behavior_policy_logit_ = np.ones((n_rounds, n_unique_action))
        evaluation_policy_logit_ = linear_behavior_policy_logit(
            context=bandit_feedback["context"],
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
    else:
        behavior_policy_logit_ = behavior_policy_function(
            context=bandit_feedback["context"],
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        evaluation_policy_logit_ = evaluation_policy_beta * behavior_policy_logit_

    if estimator_name == "IPS":
        bandit_feedback[
            "evaluation_policy_pscore"
        ] = dataset.calc_pscore_of_basic_user_behavior_assumption(
            policy_logit_=evaluation_policy_logit_,
            action=bandit_feedback["action"],
            user_behavior_assumption="standard",
            epsilon=epsilon,
        )
    else:
        bandit_feedback["evaluation_policy_pscore"] = None

    if estimator_name == "RIPS":
        bandit_feedback[
            "evaluation_policy_pscore_cascade"
        ] = dataset.calc_pscore_of_basic_user_behavior_assumption(
            policy_logit_=evaluation_policy_logit_,
            action=bandit_feedback["action"],
            user_behavior_assumption="cascade",
            epsilon=epsilon,
        )
    else:
        bandit_feedback["evaluation_policy_pscore_cascade"] = None

    if estimator_name == "IIPS":
        bandit_feedback[
            "evaluation_policy_pscore_item_position"
        ] = dataset.calc_pscore_of_basic_user_behavior_assumption(
            policy_logit_=evaluation_policy_logit_,
            action=bandit_feedback["action"],
            user_behavior_assumption="independent",
            epsilon=epsilon,
        )
    else:
        bandit_feedback["evaluation_policy_pscore_item_position"] = None

    if estimator_name == "AIPS (true)":
        bandit_feedback[
            "pscore_given_user_behavior_model"
        ] = dataset.calc_pscore_given_reward_structure(
            policy_logit_=behavior_policy_logit_,
            action=bandit_feedback["action"],
            reward_structure=bandit_feedback["reward_structure"],
        )
        bandit_feedback[
            "evaluation_policy_pscore_given_user_behavior_model"
        ] = dataset.calc_pscore_given_reward_structure(
            policy_logit_=evaluation_policy_logit_,
            action=bandit_feedback["action"],
            reward_structure=bandit_feedback["reward_structure"],
            epsilon=epsilon,
        )
    else:
        bandit_feedback["pscore_given_user_behavior_model"] = None
        bandit_feedback["evaluation_policy_pscore_given_user_behavior_model"] = None

    # on-policy
    bandit_feedback["on_policy_policy_value"] = dataset.calc_on_policy_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=bandit_feedback["context"],
        epsilon=epsilon,
    )

    # ground-truth (or large size on-policy policy value estimate)
    if evaluation == "ground_truth":
        bandit_feedback[
            "ground_truth_policy_value"
        ] = dataset.calc_ground_truth_policy_value(
            evaluation_policy_logit_=evaluation_policy_logit_,
            context=bandit_feedback["context"],
            epsilon=epsilon,
        )
    else:
        bandit_feedback["ground_truth_policy_value"] = bandit_feedback[
            "on_policy_policy_value"
        ]

    with open(path_bandit_feedback, "wb") as f:
        pickle.dump(bandit_feedback, f)

    return dataset, bandit_feedback, behavior_policy_function


def evaluate_estimators(
    estimator_name: str,
    n_rounds: int,
    len_list: int,
    n_unique_action: int,
    dim_context: int,
    reward_type: str,
    reward_structure_type: str,
    interaction_function: str,
    decay_function: str,
    behavior_policy: str,
    evaluation_policy_beta: float,
    epsilon: Optional[float],
    is_factorizable: bool,
    candidate_reward_structures: List[np.ndarray],
    candidate_reward_structure_num_random_actions: np.ndarray,
    candidate_reward_structure_weights: np.ndarray,
    candidate_reward_structure_coefs: np.ndarray,
    user_behavior_function: str,
    user_behavior_weight_scaler: Union[str, int],
    user_behavior_interpolation_param: Union[str, int],
    is_deterministic_user_behavior: bool,
    reward_std: float,
    reward_scale: float,
    interaction_scale: float,
    normalize_interaction: bool,
    target_param: str,
    target_value: Union[int, float],
    evaluation: str,
    ubtree_hyperparams: Dict[str, int],
    random_state: int,
    base_random_state: int,
    log_dir: str,
    **kwargs,
):
    print(f"random_state={random_state}, {target_param}={target_value} started")
    start = time.time()
    # convert configurations
    reward_structure_type = f"{reward_structure_type}_{interaction_function}"

    # estimators setting
    if estimator_name == "IPS":
        ips = SlateStandardIPS(len_list=len_list, estimator_name="IPS")
        ope_estimators = [ips]

    elif estimator_name == "IIPS":
        iips = SlateIndependentIPS(len_list=len_list, estimator_name="IIPS")
        ope_estimators = [iips]

    elif estimator_name == "RIPS":
        rips = SlateRewardInteractionIPS(len_list=len_list, estimator_name="RIPS")
        ope_estimators = [rips]

    elif estimator_name == "AIPS (true)":
        aips_true = SlateAdaptiveIPS(len_list=len_list, estimator_name="AIPS (true)")
        ope_estimators = [aips_true]

    # script
    dataset, bandit_feedback, behavior_policy_function = generate_and_obtain_dataset(
        estimator_name=estimator_name,
        n_rounds=n_rounds,
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure_type=reward_structure_type,
        decay_function=decay_function,
        behavior_policy=behavior_policy,
        evaluation_policy_beta=evaluation_policy_beta,
        epsilon=epsilon,
        is_factorizable=is_factorizable,
        candidate_reward_structures=candidate_reward_structures,
        candidate_reward_structure_num_random_actions=candidate_reward_structure_num_random_actions,
        candidate_reward_structure_weights=candidate_reward_structure_weights,
        candidate_reward_structure_coefs=candidate_reward_structure_coefs,
        user_behavior_function=user_behavior_function,
        user_behavior_weight_scaler=user_behavior_weight_scaler,
        user_behavior_interpolation_param=user_behavior_interpolation_param,
        is_deterministic_user_behavior=is_deterministic_user_behavior,
        reward_std=reward_std,
        reward_scale=reward_scale,
        interaction_scale=interaction_scale,
        normalize_interaction=normalize_interaction,
        evaluation=evaluation,
        random_state=random_state,
        base_random_state=base_random_state,
        log_dir=log_dir,
    )
    n_samples = len(bandit_feedback["context"])
    estimation_dict_ = dict()

    # basic ope
    ope = SlateOffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=ope_estimators,
    )
    estimation_dict_ = ope.estimate_policy_values(
        evaluation_policy_pscore=bandit_feedback["evaluation_policy_pscore"],
        evaluation_policy_pscore_item_position=bandit_feedback[
            "evaluation_policy_pscore_item_position"
        ],
        evaluation_policy_pscore_cascade=bandit_feedback[
            "evaluation_policy_pscore_cascade"
        ],
        evaluation_policy_pscore_given_user_behavior_model=bandit_feedback[
            "evaluation_policy_pscore_given_user_behavior_model"
        ],
    )

    # estimand
    estimation_dict_["on_policy"] = bandit_feedback["on_policy_policy_value"]
    estimation_dict_["ground_truth"] = bandit_feedback["ground_truth_policy_value"]

    finish = time.time()
    print(
        f"random_state={random_state}, {target_param}={target_value} finished",
        format_runtime(start, finish),
    )
    return estimation_dict_, (finish - start)


def process(
    conf: Dict[str, Any],
    start_random_state: int,
    n_random_state: int,
    base_random_state: int,
):
    working_dir = os.getcwd()
    os.chdir("../../../")
    conf["log_dir"] = os.getcwd()
    os.chdir(working_dir)

    dim_context = conf["dim_context"]
    len_list = conf["len_list"]
    user_behavior_weight_scaler = conf["user_behavior_weight_scaler"]
    user_behavior_interpolation_param = conf["user_behavior_interpolation_param"]
    estimator_name = conf["estimator_name"]

    if user_behavior_weight_scaler == "None":
        user_behavior_weight_scaler = None
    if user_behavior_interpolation_param == "None":
        user_behavior_interpolation_param = None

    target_param = conf["target_param"]
    if target_param == "user_behavior_weight_scaler":
        conf["candidate_reward_structures"] = []
        conf["candidate_reward_structure_num_random_actions"] = []
        conf["candidate_reward_structure_weights"] = []
        conf["candidate_reward_structure_coefs"] = []

        for user_behavior_weight_scaler in conf["user_behavior_weight_scaler"]:
            (
                candidate_reward_structures,
                candidate_reward_structure_num_random_actions,
                candidate_reward_structure_weights,
                candidate_reward_structure_coefs,
            ) = diverse_reward_structure(
                len_list=len_list,
                dim_context=dim_context,
                weight_scaler=user_behavior_weight_scaler,
                weight_interpolation_param=None,
                random_state=base_random_state,
            )
            conf["candidate_reward_structures"].append(candidate_reward_structures)
            conf["candidate_reward_structure_num_random_actions"].append(
                candidate_reward_structure_num_random_actions
            )
            conf["candidate_reward_structure_weights"].append(
                candidate_reward_structure_weights
            )
            conf["candidate_reward_structure_coefs"].append(
                candidate_reward_structure_coefs
            )

    elif target_param == "user_behavior_interpolation_param":
        conf["candidate_reward_structures"] = []
        conf["candidate_reward_structure_num_random_actions"] = []
        conf["candidate_reward_structure_weights"] = []
        conf["candidate_reward_structure_coefs"] = []

        for user_behavior_interpolation_param in conf[
            "user_behavior_interpolation_param"
        ]:
            (
                candidate_reward_structures,
                candidate_reward_structure_num_random_actions,
                candidate_reward_structure_weights,
                candidate_reward_structure_coefs,
            ) = diverse_reward_structure(
                len_list=len_list,
                dim_context=dim_context,
                weight_scaler=None,
                weight_interpolation_param=user_behavior_interpolation_param,
                random_state=base_random_state,
            )
            conf["candidate_reward_structures"].append(candidate_reward_structures)
            conf["candidate_reward_structure_num_random_actions"].append(
                candidate_reward_structure_num_random_actions
            )
            conf["candidate_reward_structure_weights"].append(
                candidate_reward_structure_weights
            )
            conf["candidate_reward_structure_coefs"].append(
                candidate_reward_structure_coefs
            )

    elif target_param == "len_list":
        conf["candidate_reward_structures"] = []
        conf["candidate_reward_structure_num_random_actions"] = []
        conf["candidate_reward_structure_weights"] = []
        conf["candidate_reward_structure_coefs"] = []

        for len_list in conf["len_list"]:
            (
                candidate_reward_structures,
                candidate_reward_structure_num_random_actions,
                candidate_reward_structure_weights,
                candidate_reward_structure_coefs,
            ) = diverse_reward_structure(
                len_list=len_list,
                dim_context=dim_context,
                weight_scaler=user_behavior_weight_scaler,
                weight_interpolation_param=user_behavior_interpolation_param,
                random_state=base_random_state,
            )
            conf["candidate_reward_structures"].append(candidate_reward_structures)
            conf["candidate_reward_structure_num_random_actions"].append(
                candidate_reward_structure_num_random_actions
            )
            conf["candidate_reward_structure_weights"].append(
                candidate_reward_structure_weights
            )
            conf["candidate_reward_structure_coefs"].append(
                candidate_reward_structure_coefs
            )

    else:
        (
            conf["candidate_reward_structures"],
            conf["candidate_reward_structure_num_random_actions"],
            conf["candidate_reward_structure_weights"],
            conf["candidate_reward_structure_coefs"],
        ) = diverse_reward_structure(
            len_list=len_list,
            dim_context=dim_context,
            weight_scaler=user_behavior_weight_scaler,
            weight_interpolation_param=user_behavior_interpolation_param,
            random_state=base_random_state,
        )

    # multiprocess with different random state
    p = Pool(cpu_count() // 2)
    returns = []
    for random_state in range(start_random_state, n_random_state):
        return_ = p.apply_async(
            wrapper_evaluate_estimators, args=((conf, random_state),)
        )
        returns.append(return_)
    p.close()

    # aggregate results
    estimators_name = [
        estimator_name,
        "on_policy",
    ]
    estimators_performance = defaultdict(list)
    runtimes, target_values, random_states = [], [], []

    for return_ in returns:
        estimation_dict, runtime, target_value, random_state = return_.get()

        for estimation_dict_, runtime_, target_value_ in zip(
            estimation_dict, runtime, target_value
        ):
            for estimator_name_ in estimators_name:
                estimation_ = estimation_dict_[estimator_name_]
                estimators_performance[estimator_name].append(estimation_)

            runtimes.append(runtime_)
            target_values.append(target_value_)
            random_states.append(random_state)

    # save logs
    experiment = conf["experiment"]

    df = pd.DataFrame()
    df[f"{estimator_name}"] = runtimes
    df["target_value"] = target_values
    df["random_state"] = random_states
    df.to_csv(f"runtime_{experiment}_{estimator_name}.csv", index=False)


def wrapper_evaluate_estimators(args):
    conf, random_state = args
    conf["random_state"] = random_state
    target_param = conf["target_param"]

    estimation_dict, runtime, target_value = [], [], []
    if target_param in ["n_rounds", "len_list"]:
        for i, value in enumerate(conf[target_param]):
            conf_ = copy.deepcopy(conf)
            conf_[target_param] = int(value)
            conf_["target_value"] = value

            if target_param == "len_list":
                conf_["candidate_reward_structures"] = conf[
                    "candidate_reward_structures"
                ][i]
                conf_["candidate_reward_structure_num_random_actions"] = conf[
                    "candidate_reward_structure_num_random_actions"
                ][i]
                conf_["candidate_reward_structure_weights"] = conf[
                    "candidate_reward_structure_weights"
                ][i]
                conf_["candidate_reward_structure_coefs"] = conf[
                    "candidate_reward_structure_coefs"
                ][i]

            estimation_dict_, runtime_ = evaluate_estimators(**conf_)
            estimation_dict.append(estimation_dict_)
            runtime.append(runtime_)
            target_value.append(value)

    elif target_param in ["reward_std", "epsilon"]:
        for i, value in enumerate(conf[target_param]):
            conf_ = copy.deepcopy(conf)
            conf_[target_param] = float(value)
            conf_["target_value"] = value

            estimation_dict_, runtime_ = evaluate_estimators(**conf_)
            estimation_dict.append(estimation_dict_)
            runtime.append(runtime_)
            target_value.append(value)

    else:
        for i, value in enumerate(conf[target_param]):
            conf_ = copy.deepcopy(conf)
            conf_[target_param] = float(value)
            conf_["target_value"] = value

            conf_["candidate_reward_structures"] = conf["candidate_reward_structures"][
                0
            ]
            conf_["candidate_reward_structure_num_random_actions"] = conf[
                "candidate_reward_structure_num_random_actions"
            ][0]
            conf_["candidate_reward_structure_weights"] = conf[
                "candidate_reward_structure_weights"
            ][i]
            conf_["candidate_reward_structure_coefs"] = conf[
                "candidate_reward_structure_coefs"
            ][0]

            estimation_dict_, runtime_ = evaluate_estimators(**conf_)
            estimation_dict.append(estimation_dict_)
            runtime.append(runtime_)
            target_value.append(value)

    return estimation_dict, runtime, target_value, random_state


def assert_configuration(cfg: DictConfig):
    estimator_name = cfg.setting.estimator_name
    assert estimator_name in ["IPS", "IIPS", "RIPS", "AIPS"]

    start_random_state = cfg.setting.start_random_state
    assert isinstance(start_random_state, int) and start_random_state >= 0

    n_random_state = cfg.setting.n_random_state
    assert isinstance(n_random_state, int) and n_random_state > start_random_state

    base_random_state = cfg.setting.base_random_state
    assert isinstance(base_random_state, int) and base_random_state > 0

    n_rounds = cfg.setting.n_rounds
    if isinstance(n_rounds, ListConfig):
        for value in n_rounds:
            assert isinstance(value, int) and value > 0
    else:
        assert isinstance(n_rounds, int) and n_rounds > 0

    len_list = cfg.setting.len_list
    if isinstance(len_list, ListConfig):
        for value in len_list:
            assert isinstance(value, int) and value > 0
    else:
        assert isinstance(len_list, int) and len_list > 0

    reward_structure_type = cfg.setting.reward_structure_type
    assert reward_structure_type == "context_dependent"

    interaction_function = cfg.setting.interaction_function
    assert interaction_function in ["additive", "decay"]

    is_factorizable = cfg.setting.is_factorizable
    assert isinstance(is_factorizable, bool)

    behavior_policy = cfg.setting.behavior_policy
    assert behavior_policy in ["linear", "uniform"]

    evaluation_policy_beta = cfg.setting.evaluation_policy_beta
    assert isinstance(evaluation_policy_beta, float)

    reward_scale = cfg.setting.reward_scale
    assert isinstance(reward_scale, float) and reward_scale > 0

    interaction_scale = cfg.setting.interaction_scale
    assert isinstance(interaction_scale, float) and interaction_scale > 0

    user_behavior_weight_scaler = cfg.setting.user_behavior_weight_scaler
    if user_behavior_weight_scaler != "None":
        if isinstance(user_behavior_weight_scaler, ListConfig):
            for value in user_behavior_weight_scaler:
                assert isinstance(value, float) and 0 <= value
        else:
            assert (
                isinstance(user_behavior_weight_scaler, float)
                and 0 <= user_behavior_weight_scaler
            )

    user_behavior_interpolation_param = cfg.setting.user_behavior_interpolation_param
    if user_behavior_interpolation_param != "None":
        if isinstance(user_behavior_interpolation_param, ListConfig):
            for value in user_behavior_interpolation_param:
                assert isinstance(value, float) and 0 <= value <= 1
        else:
            assert (
                isinstance(user_behavior_interpolation_param, float)
                and 0 <= user_behavior_interpolation_param <= 1
            )

    user_behavior_function = cfg.setting.user_behavior_function
    assert user_behavior_function in ["linear"]

    target_param = cfg.setting.target_param
    assert target_param in [
        "n_rounds",
        "len_list",
        "reward_std",
        "epsilon",
        "user_behavior_weight_scaler",
        "user_behavior_interpolation_param",
    ]

    evaluation = cfg.setting.evaluation
    assert evaluation in ["ground_truth", "on_policy"]

    min_samples_leaf = cfg.ubtree_hyperparams.min_samples_leaf
    assert isinstance(min_samples_leaf, int) and min_samples_leaf > 0

    max_depth = cfg.ubtree_hyperparams.max_depth
    assert isinstance(max_depth, int) and max_depth > 0

    noise_level = cfg.ubtree_hyperparams.noise_level
    assert isinstance(noise_level, float) and 0 <= noise_level <= 1

    n_bootstrap = cfg.ubtree_hyperparams.n_bootstrap
    assert isinstance(n_bootstrap, int) and n_bootstrap > 0

    n_partition = cfg.ubtree_hyperparams.n_partition
    assert isinstance(n_partition, int) and n_partition > 0


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    assert_configuration(cfg)
    conf = {
        "experiment": cfg.setting.experiment,
        "estimator_name": "AIPS (true)"
        if cfg.setting.estimator_name == "AIPS"
        else cfg.setting.estimator_name,
        "n_rounds": cfg.setting.n_rounds,  #
        "n_unique_action": cfg.setting.n_unique_action,
        "len_list": cfg.setting.len_list,  #
        "dim_context": cfg.setting.dim_context,
        "reward_type": cfg.setting.reward_type,
        "reward_structure_type": cfg.setting.reward_structure_type,
        "interaction_function": cfg.setting.interaction_function,  #
        "decay_function": cfg.setting.decay_function,
        "behavior_policy": cfg.setting.behavior_policy,
        "evaluation_policy_beta": cfg.setting.evaluation_policy_beta,
        "epsilon": cfg.setting.epsilon if cfg.setting.epsilon != "None" else None,
        "is_factorizable": cfg.setting.is_factorizable,
        "user_behavior_weight_scaler": cfg.setting.user_behavior_weight_scaler,
        "user_behavior_interpolation_param": cfg.setting.user_behavior_interpolation_param,
        "user_behavior_function": cfg.setting.user_behavior_function,
        "is_deterministic_user_behavior": cfg.setting.is_deterministic_user_behavior,
        "reward_std": cfg.setting.reward_std,
        "reward_scale": cfg.setting.reward_scale,
        "interaction_scale": cfg.setting.interaction_scale,
        "normalize_interaction": cfg.setting.normalize_interaction,
        "target_param": cfg.setting.target_param,
        "evaluation": cfg.setting.evaluation,
        "base_random_state": cfg.setting.base_random_state,
        "ubtree_hyperparams": cfg.ubtree_hyperparams,
    }
    start_random_state = cfg.setting.start_random_state
    n_random_state = cfg.setting.n_random_state
    base_random_state = cfg.setting.base_random_state
    # script
    process(
        conf=conf,
        start_random_state=start_random_state,
        n_random_state=n_random_state,
        base_random_state=base_random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))

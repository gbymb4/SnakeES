from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from optim.base_optim import BaseOptim
from utils.optimizers import Adam
from utils.policy_dict import agent_policy


class ESOpenAI(BaseOptim):
    def __init__(self, config):
        super(ESOpenAI, self).__init__()
        self.name = config["name"]
        self.sigma_init = config["sigma_init"]
        self.sigma_curr = self.sigma_init
        self.sigma_decay = config["sigma_decay"]
        self.learning_rate = config["learning_rate"]
        self.population_size = config["population_size"]
        self.weight_decay = config['weight_decay']
        self.reward_shaping = config['reward_shaping']
        self.reward_norm = config['reward_norm']

        self.epsilons = []  # save epsilons with respect to every model

        self.agent_ids = None
        self.mu_model = None
        self.optimizer = None

    # Init policies of θ_t and (θ_t + σϵ_i)
    def init_population(self, policy: torch.nn.Module, env):
        # first, init θ_t
        self.agent_ids = env.get_agent_ids()
        policy.norm_init()
        self.mu_model = policy
        # TODO: set RL schedular
        self.optimizer = Adam(self.mu_model, self.learning_rate)

        # second, init (θ_t + σϵ_i)
        perturbations = self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)
        return perturbations

    def init_perturbations(self, agent_ids: list, mu_model: torch.nn.Module, sigma, pop_size):
        perturbations = []  # policy F_i
        self.epsilons = []  # epsilons list

        # add mu model to perturbations for future evaluation
        perturbations.append(agent_policy(agent_ids, mu_model))

        # init zero_eps for saving future eps
        zero_eps = deepcopy(mu_model)
        zero_eps.zero_init()
        zero_eps_param_lst = zero_eps.get_param_list()
        self.epsilons.append(zero_eps)

        # a loop of producing perturbed policy
        for _num in range(pop_size):
            perturbed_policy = deepcopy(mu_model)
            perturbed_policy.set_policy_id(_num)
            perturbed_policy_param_lst = perturbed_policy.get_param_list()

            epsilon_policy = deepcopy(mu_model)
            eps_policy_param_list = deepcopy(zero_eps_param_lst)

            for eps_param, perturb_param in zip(eps_policy_param_list, perturbed_policy_param_lst):
                epsilon = np.random.normal(size=perturb_param.shape)

                eps_param += epsilon  # save epsilon for the future
                perturb_param += epsilon * sigma  # theta_t + sigma * epsilon_i

            perturbed_policy.set_param_list(perturbed_policy_param_lst)
            perturbations.append(agent_policy(agent_ids, perturbed_policy))

            epsilon_policy.set_param_list(eps_policy_param_list)
            self.epsilons.append(epsilon_policy)  # append epsilon for current generation

        return perturbations

    def next_population(self, assemble, results):
        rewards = results['rewards'].tolist()
        best_reward_sofar = max(rewards)
        rewards = np.array(rewards)

        # fitness shaping
        if self.reward_shaping:
            rewards = compute_centered_ranks(rewards)

        # normalization
        if self.reward_norm:
            r_std = rewards.std()
            rewards = (rewards - rewards.mean()) / r_std

        # init next mu model
        grad = deepcopy(self.mu_model)
        grad_param_list = grad.get_param_list()
        for grad_param in grad_param_list:
            grad_param *= 0

        update_factor = 1 / ((len(self.epsilons) - 1) * self.sigma_curr)  # epsilon -1 because parent policy is included
        update_factor *= -1.0  # adapt to minimization

        # sum of F_j * epsilon_j
        for eps_idx, eps in enumerate(self.epsilons):
            eps_param_list = eps.get_param_list()
            for grad_param, eps_param in zip(grad_param_list, eps_param_list):
                grad_param += eps_param * rewards[eps_idx]

        # multiple updating factor and sum of F_j * epsilon_j
        for grad_param in grad_param_list:
            grad_param *= update_factor

        self.optimizer.update(grad_param_list)

        perturbations = self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)

        # if self.sigma_curr >= 0.01:
        #     self.sigma_curr *= self.sigma_decay

        return perturbations, self.sigma_curr, best_reward_sofar

    def get_elite_model(self):
        return self.mu_model


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

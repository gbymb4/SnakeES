from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import multiprocessing as mp

from torch.distributions import MultivariateNormal

from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params, xavier_init, compute_centered_ranks
from assembly.assemble_rl import worker_func


class NESDeepMind(BaseOptim):
    def __init__(self, config):
        super(NESDeepMind, self).__init__()

        self.name = config["name"]
        self.sigma_init = config["sigma_init"]
        self.sigma_curr = self.sigma_init
        self.sigma_decay = config["sigma_decay"]
        self.learning_rate = config["learning_rate"]
        self.population_size = config["population_size"]
        self.weight_decay = config["weight_decay"]
        self.reward_shaping = config["reward_shaping"]
        self.reward_norm = config["reward_norm"]

        self.truncation_selection = config["truncation_selection"]
        self.top_T = config["top_T"]  # top T individuals become the parents of the next generation
        self.elite_candidate_size = config["elite_candidate_size"]
        self.addition_eval_elite_candidate = config["addition_eval_elite_candidate"]
        self.elitism = config["elitism"]

        self.agent_ids = None

        self.agent_ids = None
        self.mean = None
        self.cov = None
        self.mvn = None  # MultivariateNormal model
        self.mu_model = None
        self.optimizer = None
        self.perturbation_param = None

    def init_population(self, policy: torch.nn.Module, env):
        self.agent_ids = env.get_agent_ids()
        perturbation = []
        perturbation_param = []

        for _num in range(self.population_size):
            perturbed_policy = deepcopy(policy)
            perturbed_policy.set_policy_id(_num)
            perturbed_policy.norm_init()
            perturbation_param.append(get_flatten_params(perturbed_policy)['params'])
            perturbation.append(agent_policy(self.agent_ids, perturbed_policy))

        # Calculate the init mean, covariance matrix, and multivariate normal model
        self.perturbation_param = np.array(perturbation_param)
        self.mean = torch.from_numpy(np.mean(a=self.perturbation_param, axis=0))
        self.cov = torch.from_numpy(np.ma.cov(x=self.perturbation_param, rowvar=True))
        self.mvn = MultivariateNormal(loc=self.mean, covariance_matrix=self.cov)

        # Init optimizer
        self.optimizer = torch.optim.Adam([self.mean, self.cov], lr=self.learning_rate)

        return perturbation

    def next_population(self, assemble, results):

        rewards = results['rewards'].tolist()
        best_reward_this_generation = max(rewards)
        rewards = np.array(rewards)

        # fitness shaping
        if self.reward_shaping:
            rewards = compute_centered_ranks(rewards)

        # normalization
        if self.reward_norm:
            r_std = rewards.std()
            rewards = (rewards - rewards.mean()) / r_std

        # update mean and cov based on gradient
        self.optimizer.zero_grad()
        loss = -torch.mean(self.mvn.log_prob(torch.from_numpy(self.perturbation_param)) * torch.from_numpy(rewards))
        loss.backward()
        self.optimizer.step()

        # sample new perturbation based on mean and cov
        perturbation_next = self.mvn.sample(self.population_size)

        # create new population based on sampled individuals as NN params
        for _num in range(self.population_size):
            perturbed_policy = deepcopy(policy)
            perturbed_policy.set_policy_id(_num)
            perturbed_policy.norm_init()
            self.perturbation_param.append(get_flatten_params(perturbed_policy)['params'])
            perturbation.append(agent_policy(self.agent_ids, perturbed_policy))

        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return self.perturbation, self.sigma_curr, best_reward_this_generation

    def get_elite_model(self):
        return self.elite_model['0']

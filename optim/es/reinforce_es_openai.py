from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from optim.es.es_openai import ESOpenAI


class ReinforceESOpenAI(ESOpenAI):
    def __init__(self, config):
        super(ReinforceESOpenAI, self).__init__(config)
        self.reinforce_lr = config["reinforce_lr"]

    def init_population(self, policy: torch.nn.Module, env):
        return super(ReinforceESOpenAI, self).init_population(policy, env)

    def init_perturbations(self, agent_ids: list, mu_model: torch.nn.Module, sigma, pop_size):
        return super(ReinforceESOpenAI, self).init_perturbations(agent_ids, mu_model, sigma, pop_size)

    def next_population(self, assemble, results):
        rewards = results['rewards'].tolist()
        best_reward_sofar = max(rewards)
        G = results['G'].tolist()

        # normalize reward and G values
        G_np = np.array(G)
        G_norm = (G_np - G_np.mean()) / G_np.std()

        r_np = np.array(rewards)
        r_norm = (r_np - r_np.mean()) / r_np.std()

        # update r_norm based on G_norm and reinforce learning rate
        r_norm = r_norm + self.reinforce_lr * G_norm

        # init next mu model
        grad = deepcopy(self.mu_model)
        grad_param_list = grad.get_param_list()
        for grad_param in grad_param_list:
            grad_param *= 0

        # updating factor
        update_factor = 1 / ((len(self.epsilons) - 1) * self.sigma_curr)  # epsilon -1 because parent policy is included
        update_factor *= -1.0  # adapt to minimization

        # sum of F_j * epsilon_j
        for eps_idx, eps in enumerate(self.epsilons):
            eps_param_list = eps.get_param_list()
            for grad_param, eps_param in zip(grad_param_list, eps_param_list):
                grad_param += eps_param * r_norm[eps_idx]

        # multiple updating factor and sum of F_j * epsilon_j
        for grad_param in grad_param_list:
            grad_param *= update_factor

        self.optimizer.update(grad_param_list)

        perturbations = self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)

        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return perturbations, self.sigma_curr, best_reward_sofar

    def get_elite_model(self):
        return self.mu_model

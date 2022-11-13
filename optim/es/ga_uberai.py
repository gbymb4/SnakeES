from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import multiprocessing as mp

from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params, xavier_init
from assembly.assemble_rl import worker_func
import assembly.assemble_rl


class GAUberAI(BaseOptim):
    def __init__(self, config):
        super(GAUberAI, self).__init__()

        self.name = config["name"]
        self.sigma_init = config["sigma_init"]
        self.sigma_curr = self.sigma_init
        self.sigma_decay = config["sigma_decay"]
        self.learning_rate = config["learning_rate"]
        self.population_size = config["population_size"]
        self.weight_decay = config["weight_decay"]

        self.truncation_selection = config["truncation_selection"]
        self.top_T = config["top_T"]  # top T individuals become the parents of the next generation
        self.elite_candidate_size = config["elite_candidate_size"]
        self.addition_eval_elite_candidate = config["addition_eval_elite_candidate"]
        self.elitism = config["elitism"]

        self.agent_ids = None

        self.perturbation = []
        self.elite_model = None
        self.elite_candidates = []
        self.g_counter = 0

    def init_population(self, policy: torch.nn.Module, env):
        self.agent_ids = env.get_agent_ids()

        for _num in range(self.population_size):
            perturbed_policy = deepcopy(policy)
            perturbed_policy.set_policy_id(_num)
            # perturbed_policy.apply(xavier_init)
            # perturbed_policy.zero_init()
            perturbed_policy.norm_init()
            self.perturbation.append(agent_policy(self.agent_ids, perturbed_policy))

        return self.perturbation

    def next_population(self, assemble, results):

        # Sorting population based on fitness
        results_df = results.sort_values(by=['rewards'], ascending=False)

        # Construct elite candidates
        if self.g_counter == 0:
            elite_policy_ids = results_df.head(self.elite_candidate_size)['policy_id'].tolist()
            self.elite_candidates = [self.perturbation[idx] for idx in elite_policy_ids]
            self.g_counter += 1
        else:
            elite_policy_ids = results_df.head(self.elite_candidate_size - 1)['policy_id'].tolist()
            self.elite_candidates = [self.perturbation[idx] for idx in elite_policy_ids]
            self.elite_candidates.append(self.elite_model)

        # Set elite based on extra evaluation for more robust selection
        robust_elite_id, best_reward_sofar = self.extra_eval(assemble, self.elite_candidates)

        if robust_elite_id != -1:
            self.elite_model = agent_policy(self.agent_ids, self.perturbation[robust_elite_id]['0'])  # fixme: deepcopy

        # results.append({'policy_id': -1, 'rewards': best_reward_sofar}, ignore_index=True)
        results.loc[-1] = [-1, best_reward_sofar]
        results.index = results.index + 1
        self.elite_model['0'].set_policy_id(-1)

        # Only include elite once
        isContained = False
        ind_elite = get_flatten_params(self.elite_model['0'])
        for model in self.perturbation:
            ind = get_flatten_params(model['0'])
            if np.array_equal(ind_elite['params'], ind['params']):
                isContained = True
                break
        if not isContained:
            self.perturbation.append(self.elite_model)

        # construct new population based on truncation_selection of the new population (that combines mutated and elite)
        perturbation_next = []
        if self.truncation_selection:
            # truncated list
            truncated_parent_ids = results_df.head(self.top_T)['policy_id'].tolist()
            truncated_parents = [agent_policy(self.agent_ids, self.perturbation[idx]['0']) for idx in
                                 truncated_parent_ids]

            for _num in range(self.population_size):
                model = deepcopy(self.elite_model)
                # select a parent from T number of truncated parents
                k = np.random.random_integers(0, self.top_T - 1)
                # mutate this parent by  (θ_k + σϵ_k)
                indi_k = get_flatten_params(truncated_parents[k]['0'])
                indi_k_mutated = indi_k['params'] + self.sigma_curr * np.random.randn(len(indi_k['params']))
                set_flatten_params(indi_k_mutated, indi_k['lengths'], model['0'])
                # update policy id
                model['0'].set_policy_id(_num)
                perturbation_next.append(model)

        self.perturbation = perturbation_next
        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return self.perturbation, self.sigma_curr, best_reward_sofar

    def extra_eval(self, assemble, elite_candidates):
        p = mp.Pool(assemble.processor_num)
        arguments = [(indi, assemble.env, assemble.optim, self.addition_eval_elite_candidate, assemble.ob_rms_mean,
                      assemble.ob_rms_std,
                      assemble.processor_num, 0, assemble.args, assemble.config) for indi in elite_candidates]
        if assemble.processor_num > 1:
            results_robust = p.map(worker_func, arguments)
        else:
            results_robust = [worker_func(arg) for arg in arguments]

        p.close()

        robust_elite = pd.DataFrame(results_robust).sort_values(by=['rewards'], ascending=False).head(1)
        robust_elite_id = robust_elite['policy_id'].tolist()[0]
        robust_elite_reward = robust_elite['rewards'].tolist()[0]
        return robust_elite_id, robust_elite_reward

    def get_elite_model(self):
        return self.elite_model['0']

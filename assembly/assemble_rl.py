import multiprocessing as mp
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pickle
import yaml

from assembly.base_assemble import BaseAssembleRL
from utils.running_mean_std import RunningMeanStd
from env.gym_openAI.simulator_gym import GymEnv

from utils.policy_dict import agent_policy
import builder


class AssembleRL(BaseAssembleRL):

    def __init__(self, args, config, env, policy, optim):
        super(AssembleRL, self).__init__()

        self.args = args
        self.config = config

        self.env = env
        self.policy = policy
        self.optim = optim

        #  settings for running
        self.running_mstd = config["optim"]['input_running_mean_std']
        if self.running_mstd:  # Init running mean and std
            if isinstance(self.env, GymEnv):
                self.ob_rms = RunningMeanStd(shape=self.env.env.observation_space.shape)
            else:
                self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean = self.ob_rms.mean
            self.ob_rms_std = np.sqrt(self.ob_rms.var)
        else:
            self.ob_rms = None
            self.ob_rms_mean = None
            self.ob_rms_std = None

        self.generation_num = config["optim"]['generation_num']
        self.processor_num = args.processor_num
        self.eval_ep_num = args.eval_ep_num

        # log settings
        self.log = args.log
        self.save_model_freq = args.save_model_freq
        self.ep5_rewards = deque(maxlen=5)
        self.save_mode_dir = None

    def train(self):
        # Init log repository
        now = datetime.now()
        curr_time = now.strftime("%Y%m%d%H%M%S%f")
        dir_lst = []
        self.save_mode_dir = f"logs/{self.env.name}/{curr_time}"
        dir_lst.append(self.save_mode_dir)
        dir_lst.append(self.save_mode_dir + "/saved_models/")
        dir_lst.append(self.save_mode_dir + "/train_performance/")
        for _dir in dir_lst:
            os.makedirs(_dir)
        # shutil.copyfile(self.args.config, self.save_mode_dir + "/profile.yaml")
        # save the running YAML as profile.yaml in the log
        with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
            yaml.dump(self.config, file)
            file.close()

        # Start with a population init
        population = self.optim.init_population(self.policy, self.env)

        if self.config['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        for g in range(self.generation_num):
            start_time = time.time()

            # start multiprocessing
            p = mp.Pool(self.processor_num)

            arguments = [(indi, self.env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                          self.processor_num, g, self.args, self.config) for indi in population]

            # start rollout works
            start_time_rollout = time.time()

            if self.processor_num > 1:
                results = p.map(worker_func, arguments)
            else:
                results = [worker_func(arg) for arg in arguments]

            p.close()

            # end rollout
            end_time_rollout = time.time() - start_time_rollout

            # start eval
            start_time_eval = time.time()
            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])

            population, sigma_curr, best_reward_per_g = self.optim.next_population(self, results_df)
            end_time_eval = time.time() - start_time_eval

            end_time_generation = time.time() - start_time

            # update best reward so far
            if self.config['optim']['maximization'] and (best_reward_per_g > best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            if (not self.config['optim']['maximization']) and (best_reward_per_g < best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            # print runtime infor
            print(
                f"episode: {g}, best reward so far: {best_reward_so_far:.4f}, best reward of the current generation: {best_reward_per_g:.4f}, sigma: {sigma_curr:.3f}, time_generation: {end_time_generation:.2f}, rollout_time: {end_time_rollout:.2f}, eval_time: {end_time_eval:.2f}"
            )

            # update mean and std every generation
            if self.running_mstd:
                if (g+1) % self.save_model_freq == 0:  # Save current ob_rms
                    save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{(g+1)}.pickle"
                    f = open(save_pth, 'wb')
                    pickle.dump(np.concatenate((self.ob_rms_mean, self.ob_rms_std)), f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()
                hist_obs = []
                hist_obs = np.concatenate(results_df['hist_obs'])
                # Update future ob_rms_mean  and  ob_rms_std
                self.ob_rms.update(hist_obs)
                self.ob_rms_mean = self.ob_rms.mean
                self.ob_rms_std = np.sqrt(self.ob_rms.var)

            if self.log:
                if self.running_mstd:
                    results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
                # return row of parent policy, i.e., policy_id = -1
                results_df = results_df.loc[results_df['policy_id'] == -1]
                with open(self.save_mode_dir + "/train_performance" + "/training_record.csv", "a") as f:
                    results_df.to_csv(f, index=False, header=False)

                elite = self.optim.get_elite_model()
                if (g+1) % self.save_model_freq == 0:
                    save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{(g+1)}.pt"
                    torch.save(elite.state_dict(), save_pth)

    def eval(self):
        # load policy from log
        self.policy.load_state_dict(torch.load(self.args.policy_path))
        # create an indiviual wrapped with agent id
        indi = agent_policy(self.env.get_agent_ids(), self.policy)
        # load runtime mean and std
        if self.running_mstd:
            with open(self.args.rms_path, "rb") as f:
                ob_rms = pickle.load(f)
                self.ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                self.ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]

        self.policy.eval()
        # use a random seed for simulator in testing setting
        g = np.random.randint(2 ** 31)

        arguments = [(indi, self.env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                      self.processor_num, g, self.args, self.config)]

        results = [worker_func(arg) for arg in arguments]

        results_df = pd.DataFrame(results)

        if self.log:
            results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
            dir_test = os.path.dirname(self.args.config) + "/test_performance"
            if not os.path.exists(dir_test):
                os.makedirs(dir_test)
            results_df.to_csv(dir_test + "/testing_record.csv", index=False, header=False, mode='a')


def worker_func(arguments):
    indi, env, optim, eval_ep_num, ob_rms_mean, ob_rms_std, processor_num, g, args, config = arguments

    if processor_num > 1:
        env = builder.build_env(config["env"])

    hist_rewards = {}  # rewards record all evals
    hist_obs = {}  # observation  record all evals
    hist_actions = {}
    obs = None
    total_reward = 0

    for ep_num in range(eval_ep_num):
        # makesure identical training instances for each ep_num over one generation
        if ep_num == 0:
            states = env.reset(g)  # we also reset random.seed and np.random.seed in env.reset
        else:
            seed = np.random.randint(2 ** 31)  # same random seed across indi
            states = env.reset(seed)

        rewards_per_eval = []
        obs_per_eval = []
        actions_per_eval = []
        done = False

        for agent_id, model in indi.items():
            model.reset()
        while not done:
            actions = {}
            for agent_id, model in indi.items():
                s = states[agent_id]["state"]
                # reshape s
                if s.ndim < 2:  # make sure ndim of state = 2
                    s = s[np.newaxis, :]
                # update s
                if ob_rms_mean is not None:
                    s = (s - ob_rms_mean) / ob_rms_std
                # feed s into model
                actions[agent_id] = model(s)
                states, r, done, _ = env.step(actions)

                rewards_per_eval.append(r)
                obs_per_eval.append(s)
                actions_per_eval.append(actions[agent_id])
                total_reward += r

                # trace observations
                if obs is None:
                    obs = states["0"]["state"]
                else:
                    obs = np.append(obs, states["0"]["state"], axis=0)

        hist_rewards[ep_num] = rewards_per_eval
        hist_obs[ep_num] = obs_per_eval
        hist_actions[ep_num] = actions_per_eval

    rewards_mean = total_reward / eval_ep_num

    # def results_produce(env_name, optim_name):
    if env.name == "WorkflowScheduling-v0" and optim.name == "reinforce_es_openai":
        dF = []

        for ei in range(eval_ep_num):
            dis_rew = discount_rewards(hist_rewards[ei])
            observation = hist_obs[ei]
            action = hist_actions[ei]
            for step in range(len(dis_rew)):
                for agent_id, model in indi.items():
                    # Part 1 start
                    s = observation[step][action[step]]
                    nn_outs_1 = model.forward_(s)
                    model.zero_grad()
                    nn_outs_1.backward()

                    nn_grad1_lst = []
                    for m in model.children():
                        nn_grad1_lst.append(m.weight.grad.cpu().detach().numpy())
                        nn_grad1_lst.append(m.bias.grad.cpu().detach().numpy())

                    # Part 2 start
                    s = observation[step]
                    nn_outs_2 = model.forward_(s)

                    ef_lst = []
                    nn_grad2_temp_lst = []

                    vms_tensor = torch.tensor_split(nn_outs_2, nn_outs_2.size()[1], dim=1)
                    for vm in vms_tensor:  # for each x in CanVM(t)
                        nn_grad2 = []
                        ef = np.exp(vm.cpu().detach().numpy().tolist()[0][0])
                        ef_lst.append(ef)
                        model.zero_grad()
                        vm.backward(retain_graph=True)  # True prevents for graph from being discarded
                        for m in model.children():
                            nn_grad2.append(np.multiply(ef, m.weight.grad.cpu().detach().numpy()))
                            nn_grad2.append(np.multiply(ef, m.bias.grad.cpu().detach().numpy()))
                        nn_grad2_temp_lst.append(nn_grad2)

                    # sum of nn_grad2_temp_lst with 1/sum(ef) factor
                    nn_grad2_lst = []
                    ef_factor = 1 / sum(ef_lst)
                    for layer_idx in range(len(
                            nn_grad2_temp_lst[0])):  # Todo: change len(nn_grad2_temp_lst[0]) to num_layer of model
                        for counter, nn_grad2 in enumerate(nn_grad2_temp_lst):
                            if counter == 0:
                                layer_grad = ef_factor * (np.copy(nn_grad2[layer_idx]))
                            else:
                                layer_grad += ef_factor * (nn_grad2[layer_idx])
                        nn_grad2_lst.append(layer_grad)

                    # Combine part1 and part2
                    if len(dF) == 0:
                        for layer_nn1, layer_nn_2 in zip(nn_grad1_lst, nn_grad2_lst):
                            dF.append(np.copy((layer_nn1 - layer_nn_2) * dis_rew[step]))
                    else:
                        for df_layer, layer_nn1, layer_nn_2 in zip(dF, nn_grad1_lst, nn_grad2_lst):
                            df_layer += np.copy((layer_nn1 - layer_nn_2) * dis_rew[step])

        # dot product
        G = 0

        for df_layer in dF:
            G += np.linalg.norm(df_layer / eval_ep_num) ** 2

        if indi['0'].policy_id == -1:
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": env.episode_info["VM_execHour"],
                    "VM_totHour": env.episode_info["VM_totHour"],
                    "VM_cost": env.episode_info["VM_cost"],
                    "SLA_penalty": env.episode_info["SLA_penalty"],
                    "missDeadlineNum": env.episode_info["missDeadlineNum"],
                    'G': G}
        else:  # we do not record detailed info  for non-parent policy
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": np.nan,
                    "VM_totHour": np.nan,
                    "VM_cost": np.nan,
                    "SLA_penalty": np.nan,
                    "missDeadlineNum": np.nan,
                    'G': G}

    if env.name == "WorkflowScheduling-v0" and optim.name == "es_openai":

        if indi['0'].policy_id == -1:
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": env.episode_info["VM_execHour"],
                    "VM_totHour": env.episode_info["VM_totHour"],
                    "VM_cost": env.episode_info["VM_cost"],
                    "SLA_penalty": env.episode_info["SLA_penalty"],
                    "missDeadlineNum": env.episode_info["missDeadlineNum"]}
        else:  # we do not record detailed info for non-parent policy
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": np.nan,
                    "VM_totHour": np.nan,
                    "VM_cost": np.nan,
                    "SLA_penalty": np.nan,
                    "missDeadlineNum": np.nan}

    if env.name == "FisheryManagement-v0":  # todo: store the step info after each step instead of each episode (i.e., move it up)
        if indi['0'].policy_id == -1:
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "actual_catch": env.step_info["actual catch"],
                    "catch_less2000": env.step_info["is catch less2000"],
                    "average_catch": env.step_info["average catch"],
                    "TACC_change_more5": env.step_info["is TACC change more5%"],
                    "TACC_change_more15": env.step_info["is TACC change more15%"],
                    "biomass_less10": env.step_info["is less 10%B0"],
                    "biomass_less20": env.step_info["is less 20%B0"],
                    "biomass_less40": env.step_info["is more 40%B0"]}
        else:  # we do not record detailed info for non-parent policy
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "actual_catch": np.nan,
                    "catch_less2000": np.nan,
                    "average_catch": np.nan,
                    "TACC_change_more5": np.nan,
                    "TACC_change_more15": np.nan,
                    "biomass_less10": np.nan,
                    "biomass_less20": np.nan,
                    "biomass_less40": np.nan}

    if ob_rms_mean is not None:
        return {'policy_id': indi['0'].policy_id, 'hist_obs': obs, 'rewards': rewards_mean}


    return {'policy_id': indi['0'].policy_id,
            'rewards': rewards_mean}

    # results_produce(env.name, optim.name)


def discount_rewards(rewards):
    gamma = 0.99  # gamma: discount factor in rl
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

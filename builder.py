from assembly.assemble_rl import AssembleRL
from env.gym_openAI.simulator_gym import GymEnv
from env.workflow_scheduling_v2.simulator_wf import WFEnv
from env.Fishery.simulator_fishery import FisheryEnv
from env.snake.snake_env import SnakeEnv
from optim.es.es_openai import ESOpenAI
from optim.es.ga_uberai import GAUberAI
from optim.es.nes_deepmind import NESDeepMind
from optim.es.reinforce_es_openai import ReinforceESOpenAI
from policy.gym_model import GymPolicy
from policy.wf_model import WFPolicy
from policy.snake_model import SnakePolicy
from utils.utils import get_state_num, get_action_num, is_discrete_action


class Builder:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.env = None
        self.policy = None
        self.optim = None

    def build(self):
        env = build_env(self.config["env"])
        self.config["policy"]["discrete_action"] = is_discrete_action(env)
        self.config["policy"]["state_num"] = get_state_num(env)
        self.config["policy"]["action_num"] = get_action_num(env)
        policy = build_policy(self.config["policy"])
        optim = build_optim(self.config["optim"])
        return AssembleRL(self.args, self.config, env, policy, optim)


def build_env(config):
    env_name = config["name"]
    if env_name in ["LunarLanderContinuous-v2", "CartPole-v0"]:
        return GymEnv(env_name, config)
    elif env_name == "WorkflowScheduling-v0":
        return WFEnv(env_name, config)
    elif env_name == "FisheryManagement-v0":
        return FisheryEnv(env_name, config)
    elif env_name == "Snake-v0":
        return SnakeEnv(env_name, config)
    else:
        raise AssertionError(f"{env_name} doesn't support, please specify supported a env in yaml.")


def build_policy(config):
    model_name = config["name"]
    if model_name == "model_rnn":
        return GymPolicy(config)
    elif model_name == "model_workflow":
        return WFPolicy(config)
    elif model_name == "model_snake":
        return SnakePolicy(config)
    else:
        raise AssertionError(f"{model_name} doesn't support, please specify supported a model in yaml.")


def build_optim(config):
    optim_name = config["name"]
    if optim_name == "es_openai":
        return ESOpenAI(config)
    elif optim_name == "reinforce_es_openai":
        return ReinforceESOpenAI(config)
    elif optim_name == "ga_uberai":
        return GAUberAI(config)
    elif optim_name == "nes_deepmind":
        return NESDeepMind(config)
    else:
        raise AssertionError(f"{optim_name} doesn't support, please specify supported a optim in yaml.")

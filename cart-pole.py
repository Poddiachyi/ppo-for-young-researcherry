import numpy as np
import json
import os
import argparse

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


# default config path
agent_config_path = './configs/ppo.json'
network_spec_path = './configs/lstm.json'

parser = argparse.ArgumentParser()

parser.add_argument('--visualize', default=True, type=bool)
parser.add_argument('--agent-config', default=agent_config_path, help="Agent configuration file")
parser.add_argument('--network-spec', default=network_spec_path, help="Network specification file")
parser.add_argument('--n-episodes', default=3000, type=int)
parser.add_argument('--save-models', default=True, type=bool)
parser.add_argument('--save-interval', default=500, type=int)
parser.add_argument('--load-model', default=False, type=bool)
parser.add_argument('--load-model-eps', default=500, type=int, help='loading model from episode')

args = parser.parse_args()

# Create an OpenAIgym environment.
environment = OpenAIGym('CartPole-v0', visualize=args.visualize)

# load config
with open(args.agent_config, 'r') as f:
    agent_config = json.load(f)

with open(args.network_spec, 'r') as f:
    network_spec = json.load(f)

# create agent
agent = Agent.from_spec(
    spec=agent_config,
    kwargs=dict(
        states=environment.states,
        actions=environment.actions,
        network=network_spec,
    )
)

if args.load_model:
    load_dir = './saved_models/'
    if os.path.exists(load_dir):
        model_name = 'model_ep_{}'.format(args.load_model_eps)
        to_load = os.path.join(load_dir, model_name)
        print('Loading model', to_load)
        agent.restore_model(to_load)

# Create the runner
runner = Runner(agent=agent, environment=environment)

# Callback function printing episode statistics and saving model
def episode_finished(r):
    if r.episode % 10 == 0:
        print('Episode: {}. Reward: {}'.format(r.episode, r.episode_rewards[-1]))

        # some long run statistics
        if r.episode >= 100:
            print('Avg of last 100 rewards: {}'.format(round(sum(r.episode_rewards[-100:]) / 100, 5)))

        print('------------------------------') # for beauty

    # save model
    if args.save_models and (r.episode % args.save_interval == 0):
        print('Saving')
        path = './saved_models/model_ep_{}/model'.format(r.episode)
        print(path)
        r.agent.save_model(path)



    return True # if False then finish training. e.g. when plateau reached

# Start learning
runner.run(episodes=args.n_episodes, max_episode_timesteps=3000, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)

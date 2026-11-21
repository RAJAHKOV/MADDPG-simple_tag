import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpe2 import simple_adversary_v3, simple_spread_v3
import simple_tag_boundary as simple_tag_v3

from MADDPG import MADDPG, resolve_device


def get_env(env_name, ep_len=25, render_mode=None, *,
            boundary_mode=simple_tag_v3.BOUNDARY_MODE,
            boundary_range=simple_tag_v3.BOUNDARY_RANGE,
            boundary_penalty=simple_tag_v3.BOUNDARY_PENALTY,
            num_obstacles=simple_tag_v3.NUM_OBSTACLES):
    #"""create environment and get observation and action dimension of each agent in this environment"""
    environments = {
        'simple_adversary_v3': simple_adversary_v3,
        'simple_spread_v3': simple_spread_v3,
        'simple_tag_v3': simple_tag_v3,
    }
    if env_name not in environments:
        raise ValueError(f'Unknown environment: {env_name}')
    env_kwargs = dict(max_cycles=ep_len, render_mode=render_mode, dynamic_rescaling=False)
    if env_name == 'simple_tag_v3':
        simple_tag_v3.validate_obstacles(num_obstacles)
        env_kwargs.update(boundary_mode=boundary_mode, boundary_range=boundary_range,
                          boundary_penalty=boundary_penalty, num_obstacles=num_obstacles)
    new_env = environments[env_name].parallel_env(**env_kwargs)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v3', help='name of the env',
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=50000,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.95)')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    parser.add_argument('--device', default='auto', help='auto (prefer GPU), cpu, cuda, or cuda:N')
    simple_tag_v3.add_boundary_arguments(parser)
    args = parser.parse_args()
    if args.episode_num < 1 or args.episode_length < 1 or args.learn_interval < 1:
        parser.error('episode_num, episode_length, and learn_interval must be positive')
    if args.batch_size < 1 or args.buffer_capacity < args.batch_size or args.random_steps < 0:
        parser.error('require 1 <= batch_size <= buffer_capacity and random_steps >= 0')
    if args.env_name == 'simple_tag_v3':
        try:
            simple_tag_v3.validate_boundary(args.boundary_mode, args.boundary_range, args.boundary_penalty)
            simple_tag_v3.validate_obstacles(args.num_obstacles)
        except ValueError as exc:
            parser.error(str(exc))
    device = resolve_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f'Training device: {device} ({device_name})', flush=True)

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    run_number = max((int(name) for name in os.listdir(env_dir) if name.isdigit()), default=0) + 1
    result_dir = os.path.join(env_dir, str(run_number))
    os.makedirs(result_dir)

    env, dim_info = get_env(args.env_name, args.episode_length,
                            boundary_mode=args.boundary_mode, boundary_range=args.boundary_range,
                            boundary_penalty=args.boundary_penalty, num_obstacles=args.num_obstacles)
    with open(os.path.join(result_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir, device=device)
    print(f'Results directory: {os.path.abspath(result_dir)}', flush=True)
    print(f'Discount factor gamma: {args.gamma}', flush=True)
    if args.env_name == 'simple_tag_v3':
        print(f'Obstacles: {args.num_obstacles}', flush=True)
        print(f'Boundary: x/y [-{args.boundary_range:g}, {args.boundary_range:g}]; '
              f'mode={args.boundary_mode}; touch penalty={args.boundary_penalty:g}', flush=True)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in range(args.episode_num):
        obs, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Merge the two episode-ending signals from the current parallel API.
            done = {
                agent: terminated[agent] or truncated[agent]
                for agent in terminated
            }
            # env.render()
            maddpg.add(obs, action, reward, next_obs, done)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if (step >= args.random_steps and step % args.learn_interval == 0
                    and all(len(buffer) >= args.batch_size for buffer in maddpg.buffers.values())):
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

    maddpg.save(episode_rewards)  # save model
    env.close()


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(min(window - 1, len(arr))):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
    plt.close(fig)

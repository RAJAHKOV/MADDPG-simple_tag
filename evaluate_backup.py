import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MADDPG import MADDPG
from main import get_env
import simple_tag_boundary as simple_tag_v3
from pathlib import Path
from tag_metrics import TagEpisodeMetrics, save_tag_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v3', help='name of the env',
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('folder', type=str, help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')
    parser.add_argument('--device', default='auto', help='auto (prefer GPU), cpu, cuda, or cuda:N')
    simple_tag_v3.add_boundary_arguments(parser, defaults=False)

    args = parser.parse_args()
    if args.episode_num < 1 or args.episode_length < 1:
        parser.error('episode-num and episode-length must be positive')

    model_dir = os.path.join('./results', args.env_name, args.folder)
    assert os.path.exists(model_dir)
    try:
        simple_tag_v3.resolve_boundary_arguments(args, model_dir)
    except ValueError as exc:
        parser.error(str(exc))
    gif_dir = os.path.join(model_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env, dim_info = get_env(args.env_name, args.episode_length, render_mode='rgb_array',
                            boundary_mode=args.boundary_mode, boundary_range=args.boundary_range,
                            boundary_penalty=args.boundary_penalty, num_obstacles=args.num_obstacles)
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model.pt'), device=args.device)
    print(f'Evaluation device: {maddpg.device}')

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    metric_records = []
    for episode in range(args.episode_num):
        states, infos = env.reset()
        metrics = TagEpisodeMetrics(env.unwrapped.world) if args.env_name == 'simple_tag_v3' else None
        step = 0
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        while env.agents:  # interact with the env for an episode
            actions = maddpg.select_action(states)
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            step += 1
            if metrics is not None:
                metrics.update(env.unwrapped.world, env.unwrapped.scenario, step)
            frame_list.append(Image.fromarray(env.render()))
            states = next_states

            for agent_id, reward in rewards.items():  # update reward
                agent_reward[agent_id] += reward

        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent_id, reward in agent_reward.items():
            episode_rewards[agent_id][episode] = reward
            message += f'{agent_id}: {reward:>4f}; '
        print(message)
        if metrics is not None:
            metric_records.append(metrics.result(episode + 1, step))
            summary = save_tag_metrics(Path(model_dir), metric_records, dict(
                boundary_mode=args.boundary_mode, boundary_range=args.boundary_range,
                boundary_penalty=args.boundary_penalty, num_obstacles=args.num_obstacles,
                episode_length=args.episode_length))
        # save gif
        frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                           save_all=True, append_images=frame_list[1:],
                           duration=1000 / env.metadata.get('render_fps', 30), loop=0)

    env.close()
    if metric_records:
        print(f'Tag metrics: {summary}')
    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve {args.env_name} {total_files - 3}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))
    plt.close(fig)

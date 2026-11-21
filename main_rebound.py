import argparse
import os
import types

import matplotlib.pyplot as plt
import numpy as np
from mpe2 import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from mpe2._mpe_utils.core import World

from MADDPG import MADDPG

from tqdm import tqdm


def _make_bounded_integrate_state(bound, mode='bounce'):
    """返回一个替代 World.integrate_state 的函数，在物理积分后做边界反弹/钳制。

    这样做的好处：反弹发生在环境计算观测、奖励、done 之前，
    next_obs 与真实状态一致，不会污染回放池。
    """
    def integrate_state(self, p_force):
        # 先执行原生 World.integrate_state（维护 p_pos_prev 等）
        World.integrate_state(self, p_force)

        for entity in self.entities:
            if not entity.movable:          # 跳过障碍物等不可动物体
                continue
            limit = bound - entity.size      # 考虑实体半径，避免“半个身子”出界
            for axis in range(self.dim_p):
                position = entity.state.p_pos[axis]
                if abs(position) >= limit:
                    normal = 1.0 if position >= 0 else -1.0
                    entity.state.p_pos[axis] = normal * limit
                    # 只有速度仍朝墙外时才反弹，避免“粘墙”反复触发
                    if entity.state.p_vel[axis] * normal > 0:
                        if mode == 'bounce':
                            entity.state.p_vel[axis] *= -1
                        else:
                            entity.state.p_vel[axis] = 0.0
    return integrate_state


def get_env(env_name, ep_len=25, bound=1.5, mode='bounce'):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(max_cycles=ep_len, num_obstacles=0)

    new_env.reset()

    # ===== 给 simple_tag 的 world 注入边界反弹逻辑 =====
    if env_name == 'simple_tag_v3':
        world = new_env.unwrapped.world
        world.boundary_range = bound
        world.boundary_mode = mode
        # 绑定到实例，避免影响同一进程内其它 simple_tag 环境
        world.integrate_state = types.MethodType(
            _make_bounded_integrate_state(bound, mode), world
        )
    # ===================================================

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
    parser.add_argument('--episode_num', type=int, default=5000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=100, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=50000,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    parser.add_argument('--boundary_range', type=float, default=1.5,
                        help='simple_tag 方形边界半宽（默认 1.5）')
    parser.add_argument('--boundary_mode', type=str, default='bounce',
                        choices=['bounce', 'clamp'],
                        help='边界处理方式：bounce 反弹 / clamp 贴边并清零速度')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args.env_name, args.episode_length,
                            bound=args.boundary_range, mode=args.boundary_mode)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in tqdm(range(args.episode_num), desc="Training"):
        obs, infos = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            # 边界反弹已在 env 内部的 world.integrate_state 中完成，
            # 这里不再手动修改位置/速度，next_obs 与真实状态一致。

            # 1. 获取所有智能体（即使有些已经死了，也从环境初始列表里拿）
            all_agents = env.possible_agents

            # 2. 安全地构建 done 字典，防止 KeyError
            done = {}
            for agent in all_agents:
                if agent not in terminated:
                    done[agent] = True
                else:
                    done[agent] = terminated[agent] or truncated[agent]

            # 3. 确保 next_obs 和 reward 也包含所有智能体的键
            for agent in all_agents:
                if agent not in next_obs:
                    next_obs[agent] = obs[agent]  # 死了就保持原位
                if agent not in reward:
                    reward[agent] = 0.0

            maddpg.add(obs, action, reward, next_obs, done)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 1000 == 0:  # print info every 1000 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

    maddpg.save(episode_rewards)  # save model

    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward

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
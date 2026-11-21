"""Check bounded physics, rewards, capture metrics and environment APIs."""
import argparse
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pettingzoo.test import api_test, parallel_api_test

import simple_tag_boundary as tag
from tag_metrics import TagEpisodeMetrics, save_tag_metrics, summarize_tag_metrics


def place(world):
    positions = [(-0.7, -0.7), (-0.7, 0.7), (0.7, -0.7), (0.7, 0.7)]
    for agent, pos in zip(world.agents, positions):
        agent.state.p_pos[:] = pos
        agent.state.p_vel[:] = 0


def check_walls():
    for mode in ('bounce', 'punish'):
        env = tag.parallel_env(num_obstacles=0, max_cycles=10,
                               boundary_mode=mode, boundary_penalty=2)
        env.reset(seed=20)
        world, scenario = env.unwrapped.world, env.unwrapped.scenario
        for target in world.agents:
            limit = 1.5 - target.size
            for axis in (0, 1):
                for sign in (-1, 1):
                    for corner in (False, True):
                        env.reset(seed=20)
                        place(world)
                        axes = (0, 1) if corner else (axis,)
                        for a in axes:
                            target.state.p_pos[a] = sign * (limit - 0.01)
                            target.state.p_vel[a] = sign * 0.4
                        if not corner:
                            target.state.p_vel[1 - axis] = 0.2
                        obs, rewards, _, _, _ = env.step({a: 0 for a in env.agents})
                        for a in axes:
                            np.testing.assert_allclose(target.state.p_pos[a], sign * limit)
                            expected = -sign * 0.3 if mode == 'bounce' else 0
                            np.testing.assert_allclose(target.state.p_vel[a], expected, atol=1e-10)
                        if not corner:
                            np.testing.assert_allclose(target.state.p_vel[1 - axis], 0.15, atol=1e-10)
                        assert rewards[target.name] == (-2 if mode == 'punish' else 0)
                        assert all(rewards[a.name] == 0 for a in world.agents if a is not target)
                        np.testing.assert_allclose(obs[target.name][:2], target.state.p_vel, atol=1e-7)
                        np.testing.assert_allclose(obs[target.name][2:4], target.state.p_pos, atol=1e-7)
                        _, rewards, _, _, _ = env.step({a: 0 for a in env.agents})
                        assert rewards[target.name] == (-2 if mode == 'punish' else 0)
                        target.state.p_pos[:] = [0, 0]
                        target.state.p_vel[:] = 0
                        _, rewards, _, _, _ = env.step({a: 0 for a in env.agents})
                        assert rewards[target.name] == 0

            # Exact contact with outward action reflects; inward motion is allowed.
            env.reset(seed=20)
            place(world)
            target.state.p_pos[:] = [limit, 0]
            target.state.p_vel[:] = [0, 0.2]
            env.step({a: (2 if a == target.name else 0) for a in env.agents})
            expected = -target.accel * world.dt if mode == 'bounce' else 0
            np.testing.assert_allclose(target.state.p_vel[0], expected, atol=1e-10)
            place(world)
            target.state.p_pos[:] = [-limit, 0]
            target.state.p_vel[:] = [0.4, 0]
            env.step({a: 0 for a in env.agents})
            assert target.state.p_vel[0] > 0
            assert target.state.p_pos[0] > -limit

        # Original collision rewards stack with each individual's wall penalty.
        place(world)
        adv, prey = world.agents[0], world.agents[-1]
        adv.state.p_pos[:] = [1.5 - adv.size, 0]
        prey.state.p_pos[:] = [1.5 - prey.size, 0.01]
        penalty = 2 if mode == 'punish' else 0
        assert scenario.reward(adv, world) == 10 - penalty
        assert scenario.reward(prey, world) == -10 - penalty
        assert scenario.reward(world.agents[1], world) == 10
        restored = pickle.loads(pickle.dumps(env.unwrapped))
        assert restored.boundary_mode == mode and restored.boundary_penalty == 2
        assert restored.scenario.boundary_mode == mode
        assert restored.world.boundary_range == 1.5
        restored.close()
        env.close()

        # Small custom arenas contain full entity circles at reset and after steps.
        env = tag.parallel_env(boundary_mode=mode, boundary_range=0.5, max_cycles=150)
        for seed in (1, 2):
            env.reset(seed=seed)
            while env.agents:
                for entity in env.unwrapped.world.entities:
                    assert np.all(np.abs(entity.state.p_pos) + entity.size <= 0.5 + 1e-12)
                    assert np.isfinite(entity.state.p_vel).all()
                env.step({a: env.action_space(a).sample() for a in env.agents})
        env.close()

    for kwargs in ({'boundary_range': 0}, {'boundary_range': float('nan')},
                   {'boundary_range': 0.1}, {'boundary_penalty': -1},
                   {'boundary_penalty': float('inf')}, {'boundary_mode': 'invalid'}):
        try:
            tag.raw_env(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid configuration accepted: {kwargs}')
    print('PASS: all walls, both roles, corners, tangential velocity, rewards, bounds, reset and pickle')


def check_metrics():
    env = tag.parallel_env(num_obstacles=0)
    env.reset(seed=0)
    world, scenario = env.unwrapped.world, env.unwrapped.scenario
    records = []
    for episode, capture_step in enumerate((None, 1, 5), 1):
        metric = TagEpisodeMetrics(world)
        for step in range(1, 6):
            place(world)
            if capture_step is not None and step >= capture_step:
                world.agents[-1].state.p_pos[:] = world.agents[0].state.p_pos + [0.1, 0]
            metric.update(world, scenario, step)
        records.append(metric.result(episode, 5))
    summary = summarize_tag_metrics(records)
    assert summary['adversary_capture_success_rate'] == 2 / 3
    assert summary['average_capture_steps'] == 3
    assert summary['agent_escape_success_rate'] == 1 / 3
    assert summarize_tag_metrics(records[:1])['average_capture_steps'] is None
    assert summarize_tag_metrics([])['adversary_capture_success_rate'] is None
    with TemporaryDirectory() as folder:
        save_tag_metrics(Path(folder), records, {'boundary_mode': 'bounce'})
        saved = json.loads((Path(folder) / 'evaluation_summary.json').read_text())
        assert saved['average_capture_steps'] == 3
        assert (Path(folder) / 'evaluation_metrics.csv').is_file()
        args = argparse.Namespace(env_name='simple_tag_v3', boundary_mode=None,
                                  boundary_range=None, boundary_penalty=None)
        (Path(folder) / 'run_config.json').write_text(json.dumps(dict(
            boundary_mode='punish', boundary_range=2, boundary_penalty=3)))
        tag.resolve_boundary_arguments(args, folder)
        assert (args.boundary_mode, args.boundary_range, args.boundary_penalty) == ('punish', 2, 3)
        args.boundary_mode = 'bounce'
        tag.resolve_boundary_arguments(args, folder)
        assert args.boundary_mode == 'bounce'
    env.close()

    # Team success requires all prey; escape is averaged per prey.
    env = tag.parallel_env(num_good=2, num_obstacles=0)
    env.reset(seed=0)
    world, scenario = env.unwrapped.world, env.unwrapped.scenario
    for i, agent in enumerate(world.agents):
        agent.state.p_pos[:] = [i * 0.5 - 1, 0]
    metric = TagEpisodeMetrics(world)
    prey = scenario.good_agents(world)
    prey[0].state.p_pos[:] = world.agents[0].state.p_pos + [0.1, 0]
    metric.update(world, scenario, 2)
    partial = metric.result(1, 5)
    assert not partial['capture_success'] and partial['escaped_agents'] == 1
    assert summarize_tag_metrics([partial])['agent_escape_success_rate'] == 0.5
    prey[0].state.p_pos[:] = [0.5, 1]
    prey[1].state.p_pos[:] = world.agents[0].state.p_pos + [0.1, 0]
    metric.update(world, scenario, 5)
    assert metric.result(1, 5)['capture_step'] == 5
    env.close()
    print('PASS: first/final capture, no capture, repeated capture, aggregation, multi-prey and saved config')


def run():
    check_walls()
    check_metrics()
    for mode in ('bounce', 'punish'):
        parallel_api_test(tag.parallel_env(boundary_mode=mode, max_cycles=20), num_cycles=50)
        api_test(tag.env(boundary_mode=mode, max_cycles=20), num_cycles=50)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    run()

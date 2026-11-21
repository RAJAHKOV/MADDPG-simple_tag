"""Evaluate saved MADDPG actors with a fixed camera; no training is performed."""
import argparse
import csv
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from mpe2 import simple_adversary_v3, simple_spread_v3, simple_tag_v3

from MADDPG import MADDPG


ENVIRONMENTS = {
    'simple_adversary_v3': simple_adversary_v3,
    'simple_spread_v3': simple_spread_v3,
    'simple_tag_v3': simple_tag_v3,
}


def render_fixed(world, camera_range=2.0, size=700, caption=''):
    """Map world coordinates and physical radii using one constant pixel scale.

    camera_range is the half-width of the view, centered on (0, 0).
    Off-screen objects are clipped, without moving or resizing the camera.
    This only reads the simulation and does not change collision geometry.
    """
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    scale = size / (2.0 * camera_range)
    # Draw stationary objects first so agents remain visible in front of them.
    for entity in [*world.landmarks, *world.agents]:
        x, y = np.asarray(entity.state.p_pos, dtype=float)
        radius = float(entity.size) * scale
        cx, cy = size / 2 + x * scale, size / 2 - y * scale
        if not np.isfinite([cx, cy, radius]).all():
            raise ValueError(f'Non-finite position or radius: {entity.name}')
        if cx + radius < 0 or cx - radius >= size or cy + radius < 0 or cy - radius >= size:
            continue
        color = tuple(np.clip(np.asarray(entity.color) * 200, 0, 255).astype(int))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     fill=color, outline='black', width=1)
    if caption:
        draw.text((12, 12), caption, fill='black')
    draw.text((12, size - 20), f'Fixed view: x/y [-{camera_range:g}, {camera_range:g}]', fill='black')
    return image


def next_output_dir(model_dir):
    parent = model_dir / 'evaluations'
    parent.mkdir(exist_ok=True)
    index = 1
    while True:
        candidate = parent / f'eval_{index:03d}'
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('env_name', choices=ENVIRONMENTS)
    parser.add_argument('folder', help='Training results folder number, for example 1 or 3')
    parser.add_argument('--episode-num', '--episode_num', type=int, default=10)
    parser.add_argument('--episode-length', '--episode_length', type=int, default=50)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--camera-range', type=float, default=2.0,
                        help='Fixed view half-width in world coordinates (default: 2.0)')
    parser.add_argument('--size', type=int, default=700, help='Square GIF resolution in pixels')
    parser.add_argument('--fps', type=int, default=25, help='GIF playback rate, 1 to 100')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--results-root', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--output-dir', type=Path, help='New output folder; must not already exist')
    args = parser.parse_args()
    if args.episode_num < 1 or args.episode_length < 1:
        parser.error('episode-num and episode-length must be positive')
    if not np.isfinite(args.camera_range) or args.camera_range <= 0:
        parser.error('camera-range must be finite and positive')
    if args.size < 200 or not 1 <= args.fps <= 100:
        parser.error('size must be >= 200; fps must be between 1 and 100')
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    return args


def main():
    args = parse_args()
    model_dir = (args.results_root / args.env_name / args.folder).resolve()
    model_file = model_dir / 'model.pt'
    if not model_file.is_file():
        raise FileNotFoundError(f'Model not found: {model_file}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build directly, without depending on main.py or changing the installed MPE2 code.
    env = ENVIRONMENTS[args.env_name].parallel_env(max_cycles=args.episode_length)
    try:
        env.reset(seed=args.seed)
        agents = list(env.possible_agents)
        dim_info = {
            agent: [env.observation_space(agent).shape[0], env.action_space(agent).n]
            for agent in agents
        }
        if args.output_dir:
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=False)
        else:
            output_dir = next_output_dir(model_dir)
        gif_dir = output_dir / 'gif'
        gif_dir.mkdir()
        # Load actors here so evaluation logs are written into the new output folder,
        # rather than appended to the original training log by MADDPG.load().
        model = MADDPG(dim_info, 0, 0, 0, 0, str(output_dir), device=args.device)
        checkpoint = torch.load(model_file, map_location=model.device, weights_only=True)
        if set(checkpoint) != set(agents):
            raise ValueError('Model agent names do not match the selected environment')
        for name, agent in model.agents.items():
            agent.actor.load_state_dict(checkpoint[name])
            agent.actor.eval()
        print(f'Evaluation device: {model.device}', flush=True)
        print(f'Fixed camera range: [-{args.camera_range:g}, {args.camera_range:g}]', flush=True)
        print(f'Output directory: {output_dir}', flush=True)
        world = env.unwrapped.world
        records = []
        for episode in range(1, args.episode_num + 1):
            obs, _ = env.reset(seed=args.seed + episode - 1)
            totals = {agent: 0.0 for agent in agents}
            frames = [render_fixed(world, args.camera_range, args.size,
                                   f'Episode {episode} | Step 0')]
            step = 0
            while env.agents:
                with torch.no_grad():
                    actions = model.select_action(obs)
                obs, rewards, terminated, truncated, infos = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                step += 1
                frames.append(render_fixed(world, args.camera_range, args.size,
                                           f'Episode {episode} | Step {step}'))
            frames[0].save(gif_dir / f'episode_{episode:03d}.gif', save_all=True,
                           append_images=frames[1:], duration=round(1000 / args.fps),
                           loop=0, disposal=2, optimize=False)
            record = [episode, *totals.values(), sum(totals.values())]
            records.append(record)
            print(f'episode {episode}, ' + '; '.join(f'{a}: {r:.6f}' for a, r in totals.items())
                  + f'; sum reward: {record[-1]:.6f}', flush=True)
            # Write completed episodes incrementally, independently of training rewards.pkl.
            with (output_dir / 'evaluation_rewards.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', *agents, 'sum reward'])
                writer.writerows(records)
        fig, ax = plt.subplots(figsize=(9, 5))
        for index, agent in enumerate(agents, 1):
            ax.plot([row[0] for row in records], [row[index] for row in records], label=agent)
        ax.set(xlabel='Evaluation episode', ylabel='Episode reward',
               title=f'MADDPG evaluation: {args.env_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / 'evaluation_rewards.png', dpi=150)
        plt.close(fig)
        print('Evaluation completed. No training was performed.', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()

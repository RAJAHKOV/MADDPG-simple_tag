"""Evaluate saved MADDPG actors with a fixed camera; no training is performed."""
import argparse
import colorsys
import csv
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from mpe2 import simple_adversary_v3, simple_spread_v3
import simple_tag_boundary as simple_tag_v3

from MADDPG import MADDPG
from tag_metrics import TagEpisodeMetrics, save_tag_metrics


ENVIRONMENTS = {
    'simple_adversary_v3': simple_adversary_v3,
    'simple_spread_v3': simple_spread_v3,
    'simple_tag_v3': simple_tag_v3,
}


def contact_agents(world, boundary_range=None):
    """Read contacts for this frame only, without changing simulation colors."""
    touching = set()
    if boundary_range is not None:
        for entity in world.entities:
            if entity.movable and np.any(
                    np.abs(entity.state.p_pos) + entity.size >= boundary_range - 1e-9):
                touching.add(entity.name)
    for index, first in enumerate(world.agents):
        for second in world.agents[index + 1:]:
            if (first.collide and second.collide
                    and getattr(first, 'adversary', False) != getattr(second, 'adversary', False)
                    and np.linalg.norm(first.state.p_pos - second.state.p_pos) < first.size + second.size):
                touching.update((first.name, second.name))
    return touching


def draw_scoreboard(image, totals, step_rewards):
    """Bottom-right overlay: episode returns and signed rewards for this step."""
    draw = ImageDraw.Draw(image)
    font_size = max(9, round(image.width / 50))
    lines = ['Score: total | step']
    lines.extend(f'{name}: {value:+.1f} | {float(step_rewards.get(name, 0)):+.1f}'
                 for name, value in totals.items())
    lines.append(f'Sum: {sum(totals.values()):+.1f} | {sum(step_rewards.values()):+.1f}')
    while True:
        font = ImageFont.load_default(size=font_size)
        width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 16
        if width <= image.width - 16 or font_size <= 6:
            break
        font_size -= 1
    line_height = font_size + 5
    height = line_height * len(lines) + 12
    right, bottom = image.width - 8, image.height - 27
    left, top = right - width, bottom - height
    draw.rounded_rectangle((left, top, right, bottom), radius=5,
                           fill=(248, 248, 248), outline='black', width=1)
    for index, line in enumerate(lines):
        draw.text((left + 8, top + 6 + index * line_height), line, fill='black', font=font)


def render_fixed(world, camera_range=2.0, size=700, caption='', boundary_range=None,
                 trajectories=None, totals=None, step_rewards=None, highlight_contacts=False):
    """Map world coordinates and physical radii using one constant pixel scale.

    camera_range is the half-width of the view, centered on (0, 0).
    Off-screen objects are clipped, without moving or resizing the camera.
    boundary_range optionally draws a visual square centered on the origin.
    This only reads the simulation and does not change collision geometry.
    trajectories optionally maps agent names to their positions so far; paths
    use the same world-to-pixel mapping as circles and never show future steps.
    totals and step_rewards show live scores. Contact colors are render-only.
    """
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    scale = size / (2.0 * camera_range)
    black_agents = contact_agents(world, boundary_range) if highlight_contacts else set()
    if boundary_range is not None:
        if not np.isfinite(boundary_range) or not 0 < boundary_range <= camera_range:
            raise ValueError('boundary_range must be positive and no larger than camera_range')
        half_width = boundary_range * scale
        left = max(1, size / 2 - half_width)
        right = min(size - 2, size / 2 + half_width)
        draw.rectangle((left, left, right, right), outline='black', width=3)
        label_y = left + 6 if left >= 30 else 34
        draw.text((left + 6, label_y),
                  f'Boundary: x/y [-{boundary_range:g}, {boundary_range:g}]',
                  fill='black')
    if trajectories is not None:
        for index, (name, positions) in enumerate(trajectories.items()):
            # Stable distinct colors, including multiple adversaries of one role.
            color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(
                (index * 0.61803398875) % 1.0, 0.8, 0.75))
            points = [(size / 2 + x * scale, size / 2 - y * scale) for x, y in positions]
            if len(points) > 1:
                draw.line(points, fill=color, width=max(1, size // 300))
            if points:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            # Legend identifies paths without changing the original role colors.
            legend_y = (60 + index * 14 if totals is not None
                        else size - 36 - (len(trajectories) - 1 - index) * 14)
            draw.line((12, legend_y + 5, 30, legend_y + 5), fill=color, width=2)
            draw.text((35, legend_y), name, fill=color)
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
        if entity.name in black_agents:
            color = (0, 0, 0)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     fill=color, outline='black', width=1)
    if caption:
        draw.text((12, 12), caption, fill='black')
    draw.text((12, size - 20), f'Fixed view: x/y [-{camera_range:g}, {camera_range:g}]', fill='black')
    if totals is not None:
        draw_scoreboard(image, totals, step_rewards if step_rewards is not None else {})
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
    simple_tag_v3.add_boundary_arguments(parser, defaults=False)
    parser.add_argument('--size', type=int, default=700, help='Square GIF resolution in pixels')
    parser.add_argument('--fps', type=int, default=25, help='GIF playback rate, 1 to 100')
    parser.add_argument('--trajectory-gif', action=argparse.BooleanOptionalAction, default=True,
                        help='Also save per-episode GIFs with cumulative agent paths (default: enabled; '
                             'disable with --no-trajectory-gif)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--results-root', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--output-dir', type=Path, help='New output folder; must not already exist')
    args = parser.parse_args()
    try:
        simple_tag_v3.resolve_boundary_arguments(
            args, args.results_root / args.env_name / args.folder)
    except ValueError as exc:
        parser.error(str(exc))
    if args.episode_num < 1 or args.episode_length < 1:
        parser.error('episode-num and episode-length must be positive')
    if not np.isfinite(args.camera_range) or args.camera_range <= 0:
        parser.error('camera-range must be finite and positive')
    if args.boundary_range is not None and (
            not np.isfinite(args.boundary_range) or not 0 < args.boundary_range <= args.camera_range):
        parser.error('boundary-range must be finite, positive, and <= camera-range')
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
    env_kwargs = dict(max_cycles=args.episode_length, dynamic_rescaling=False)
    if args.env_name == 'simple_tag_v3':
        env_kwargs.update(boundary_mode=args.boundary_mode, boundary_range=args.boundary_range,
                          boundary_penalty=args.boundary_penalty, num_obstacles=args.num_obstacles)
    env = ENVIRONMENTS[args.env_name].parallel_env(**env_kwargs)
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
        trajectory_dir = output_dir / 'trajectory_gif'
        if args.trajectory_gif:
            trajectory_dir.mkdir()
        # Load actors here so evaluation logs are written into the new output folder,
        # rather than appended to the original training log by MADDPG.load().
        model = MADDPG(dim_info, 0, 0, 0, 0, str(output_dir), device=args.device)
        checkpoint = torch.load(model_file, map_location=model.device, weights_only=True)
        if set(checkpoint) != set(agents):
            raise ValueError('Model agent names do not match the selected environment')
        for name, agent in model.agents.items():
            if args.env_name == 'simple_tag_v3':
                expected = agent.actor.state_dict()
                if any(key in expected and value.shape != expected[key].shape
                       for key, value in checkpoint[name].items()):
                    raise ValueError(
                        f'Model dimensions do not match simple_tag with --num-obstacles {args.num_obstacles}. '
                        'Use the obstacle count from training, or train a new model with --num-obstacles 0 '
                        'for an obstacle-free environment.')
            agent.actor.load_state_dict(checkpoint[name])
            agent.actor.eval()
        print(f'Evaluation device: {model.device}', flush=True)
        if args.env_name == 'simple_tag_v3':
            print(f'Obstacles: {args.num_obstacles}', flush=True)
        print(f'Fixed camera range: [-{args.camera_range:g}, {args.camera_range:g}]', flush=True)
        if args.boundary_range is not None:
            mode = args.boundary_mode if args.env_name == 'simple_tag_v3' else 'visual only'
            print(f'Boundary: [-{args.boundary_range:g}, {args.boundary_range:g}], mode={mode}', flush=True)
        print(f'Output directory: {output_dir}', flush=True)
        if args.trajectory_gif:
            print(f'Trajectory GIF directory: {trajectory_dir}', flush=True)
        world = env.unwrapped.world
        records = []
        metric_records = []
        settings = dict(env_name=args.env_name, boundary_mode=args.boundary_mode,
                        boundary_range=args.boundary_range, boundary_penalty=args.boundary_penalty,
                        num_obstacles=args.num_obstacles, episode_length=args.episode_length, seed=args.seed)
        for episode in range(1, args.episode_num + 1):
            obs, _ = env.reset(seed=args.seed + episode - 1)
            metrics = TagEpisodeMetrics(world) if args.env_name == 'simple_tag_v3' else None
            totals = {agent: 0.0 for agent in agents}
            rewards = {agent: 0.0 for agent in agents}

            def render_frame(step, trajectories=None):
                return render_fixed(
                    world, args.camera_range, args.size, f'Episode {episode} | Step {step}',
                    boundary_range=args.boundary_range, trajectories=trajectories,
                    totals=totals, step_rewards=rewards,
                    highlight_contacts=args.env_name == 'simple_tag_v3')

            frames = [render_frame(0)]
            trajectories = {a.name: [tuple(a.state.p_pos)] for a in world.agents}
            trajectory_frames = []
            if args.trajectory_gif:
                trajectory_frames.append(render_frame(0, trajectories))
            step = 0
            while env.agents:
                with torch.no_grad():
                    actions = model.select_action(obs)
                obs, rewards, terminated, truncated, infos = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                step += 1
                if metrics is not None:
                    metrics.update(world, env.unwrapped.scenario, step)
                frames.append(render_frame(step))
                if args.trajectory_gif:
                    for agent in world.agents:
                        trajectories[agent.name].append(tuple(agent.state.p_pos))
                    trajectory_frames.append(render_frame(step, trajectories))
            frames[0].save(gif_dir / f'episode_{episode:03d}.gif', save_all=True,
                           append_images=frames[1:], duration=round(1000 / args.fps),
                           loop=0, disposal=2, optimize=False)
            if args.trajectory_gif:
                trajectory_frames[0].save(
                    trajectory_dir / f'episode_{episode:03d}_trajectory.gif', save_all=True,
                    append_images=trajectory_frames[1:], duration=round(1000 / args.fps),
                    loop=0, disposal=2, optimize=False)
            record = [episode, *totals.values(), sum(totals.values())]
            records.append(record)
            if metrics is not None:
                metric_record = metrics.result(episode, step)
                metric_records.append(metric_record)
                summary = save_tag_metrics(output_dir, metric_records, settings)
                print(f'capture_success={metric_record["capture_success"]}; '
                      f'capture_step={metric_record["capture_step"]}; '
                      f'escaped_agents={metric_record["escaped_agents"]}', flush=True)
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
        if metric_records:
            average = summary['average_capture_steps']
            average_text = f'{average:.3f}' if average is not None else 'N/A (no successful captures)'
            print(f'Adversary capture success rate: {summary["adversary_capture_success_rate"]:.2%}\n'
                  f'Average capture steps (successful episodes): {average_text}\n'
                  f'Agent escape success rate: {summary["agent_escape_success_rate"]:.2%}', flush=True)
        print('Evaluation completed. No training was performed.', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()

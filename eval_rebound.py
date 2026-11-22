import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 这里保留你原来的导入
from main_rebound import get_env
from MADDPG import MADDPG

# ===================== 这里改参数 =====================
env_name = 'simple_tag_v3'
folder = '16' # 在这里修改为你的数据保存的文件夹名称
ep_len = 100  
model_filename = "model.pt"
num_eval_episodes = 3
camera_range = 2.0
boundary_range = 1.5   # 你的物理边界
# =====================================================

model_dir = os.path.join('./results', env_name, folder)
gif_dir = os.path.join(model_dir, 'gif')
csv_path = os.path.join(model_dir, 'evaluation_rewards.csv')
fig_path = os.path.join(model_dir, 'evaluation_rewards.png')
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)


def draw_scoreboard(image, draw, totals, step_rewards):
    """右下角面板：回合累计奖励 + 本步即时奖励"""
    font_size = max(9, round(image.width / 50))
    lines = ['Score: total | step']
    lines.extend(f'{name}: {value:+.1f} | {float(step_rewards.get(name, 0)):+.1f}'
                 for name, value in totals.items())
    lines.append(f'Sum: {sum(totals.values()):+.1f} | {sum(step_rewards.values()):+.1f}')

    while True:
        font = ImageFont.load_default(size=font_size)
        bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        width = max(b[2] for b in bboxes) + 16
        if width <= image.width - 16 or font_size <= 6:
            break
        font_size -= 1
    line_height = font_size + 5
    height = line_height * len(lines) + 12
    right, bottom = image.width - 8, image.height - 27
    left, top = right - width, bottom - height
    draw.rounded_rectangle((left, top, right, bottom), radius=5,
                           fill=(248, 248, 248), outline='black', width=1)
    for idx, line in enumerate(lines):
        draw.text((left + 8, top + 6 + idx * line_height), line, fill='black', font=font)


def render_fixed(world, camera_range=2.0, size=700, caption='', totals=None, step_rewards=None):
    """渲染，右下角带奖励面板"""
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    scale = size / (2.0 * camera_range)

    # 物理边界 [-1.5, 1.5]
    b_min = -boundary_range
    b_max = boundary_range
    left  = size / 2 + b_min * scale
    right = size / 2 + b_max * scale
    top   = size / 2 - b_max * scale
    bottom= size / 2 - b_min * scale
    draw.rectangle([left, top, right, bottom], outline="black", width=3)

    # 画地标和智能体
    for entity in [*world.landmarks, *world.agents]:
        x, y = np.asarray(entity.state.p_pos, dtype=float)
        radius = float(entity.size) * scale
        cx, cy = size / 2 + x * scale, size / 2 - y * scale
        if not np.isfinite([cx, cy, radius]).all():
            continue
        color = tuple(np.clip(np.asarray(entity.color) * 255, 0, 255).astype(int))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     fill=color, outline='black', width=1)
    if caption:
        draw.text((12, 12), caption, fill='black')

    # 画奖励面板
    if totals is not None and step_rewards is not None:
        draw_scoreboard(image, draw, totals, step_rewards)
    return image


def check_capture(world):
    """检查是否抓捕成功：adversary 和 good agent 相撞"""
    agents = world.agents
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            adv1 = getattr(a1, 'adversary', False)
            adv2 = getattr(a2, 'adversary', False)
            if adv1 != adv2:
                dist = np.linalg.norm(a1.state.p_pos - a2.state.p_pos)
                if dist < a1.size + a2.size:
                    return True
    return False


def check_escape(world):
    """是否有good agent跑出边界 ±1.5"""
    for a in world.agents:
        if not getattr(a, 'adversary', False):
            pos = a.state.p_pos
            if np.any(np.abs(pos) >= boundary_range - 1e-9):
                return True
    return False


def main():
    env, dim_info = get_env(env_name, ep_len)
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, model_filename))
    if hasattr(maddpg, 'eval'):
        try:
            maddpg.eval()
        except Exception:
            pass

    world = env.unwrapped.world
    agent_names = list(env.possible_agents)

    records = []
    metric_records = []
    capture_success_count = 0
    escape_success_count = 0
    capture_step_list = []

    for episode in range(num_eval_episodes):
        obs, infos = env.reset()
        states = obs
        frames = []
        step = 0
        done_capture = False
        done_escape = False
        capture_step = None

        totals = {name: 0.0 for name in agent_names}
        step_rewards = {name: 0.0 for name in agent_names}

        while env.agents:
            actions = maddpg.select_action(states)
            next_states, rewards, terminated, truncated, info = env.step(actions)
            states = next_states

            step_rewards = {k: float(v) for k, v in rewards.items()}
            for k, v in step_rewards.items():
                totals[k] += v

            # 只在还没抓到的时候判断抓捕
            if not done_capture and check_capture(world):
                done_capture = True
                capture_step = step
                capture_success_count += 1
                capture_step_list.append(step)
            if not done_escape and check_escape(world):
                done_escape = True
                escape_success_count += 1

            frame = render_fixed(world, camera_range=camera_range, size=700,
                                 caption=f'Ep {episode+1} | Step {step} | model.pt',
                                 totals=totals, step_rewards=step_rewards)
            frames.append(frame)
            step += 1

        total_sum = sum(totals.values())
        records.append([episode+1, *[totals[n] for n in agent_names], total_sum])
        metric_records.append({
            "episode": episode+1,
            "capture_success": done_capture,
            "capture_step": capture_step,
            "escape_success": done_escape
        })

        print(f"Episode {episode + 1} 渲染完成（{len(frames)} 帧），准备保存 GIF...")
        save_path = os.path.join(gif_dir, f'eval_{episode + 1}.gif')
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=50, loop=0, disposal=2, optimize=False)
        print(f"GIF 已保存至: {save_path}")

    # 写入 csv
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', *agent_names, 'sum reward'])
        writer.writerows(records)

    # 画奖励曲线
    fig, ax = plt.subplots(figsize=(9, 5))
    rec_arr = np.array(records)
    for idx, name in enumerate(agent_names):
        ax.plot(rec_arr[:, 0], rec_arr[:, 1+idx], label=name)
    ax.set(xlabel='Evaluation episode', ylabel='Episode reward', title='MADDPG simple_tag evaluation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # 汇总指标
    capture_rate = capture_success_count / num_eval_episodes
    escape_rate = escape_success_count / num_eval_episodes
    avg_capture_step = np.mean(capture_step_list) if capture_step_list else None
    print("\n===== Evaluation Summary =====")
    print(f"Total episodes: {num_eval_episodes}")
    print(f"Capture success rate: {capture_rate:.2%}")
    print(f"Escape success rate: {escape_rate:.2%}")
    if avg_capture_step is not None:
        print(f"Average capture step (successful): {avg_capture_step:.2f}")
    else:
        print("Average capture step (successful): N/A (no capture)")

    env.close()


if __name__ == '__main__':
    main()

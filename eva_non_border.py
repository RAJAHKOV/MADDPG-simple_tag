import os
import torch
import numpy as np
from PIL import Image, ImageDraw
#1
#from main_mpe2 import get_env
#2
#from main_penality import get_env
#3（nope）4
#from main_penality_para import get_env

#6,7，8,9，10,11,12,13,15
from main_penality_append import get_env
from MADDPG import MADDPG

# ===================== 这里改参数 =====================
env_name = 'simple_tag_v3'
folder = '15'
ep_len = 100         # 和训练时 episode_length 保持一致
model_filename = "model.pt"
num_eval_episodes = 3
camera_range = 2.0
# =====================================================

model_dir = os.path.join('./results', env_name, folder)
gif_dir = os.path.join(model_dir, 'gif')
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)


def render_fixed(world, camera_range=2.0, size=700, caption=''):
    """完全复刻手绘渲染，无头服务器运行，粗边界 [-1.5,1.5]"""
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    scale = size / (2.0 * camera_range)

    # 物理边界固定 [-1.5, 1.5]
    b_min = -1.5
    b_max = 1.5
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
    return image


def main():
    env, dim_info = get_env(env_name, ep_len)
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, model_filename))
    if hasattr(maddpg, 'eval'):
        try:
            maddpg.eval()
        except Exception:
            pass

    world = env.unwrapped.world

    for episode in range(num_eval_episodes):
        obs, infos = env.reset()
        states = obs
        frames = []
        step = 0
        while env.agents:
            actions = maddpg.select_action(states)
            next_states, rewards, terminated, truncated, info = env.step(actions)
            states = next_states

            frame = render_fixed(world, camera_range=camera_range, size=700,
                                 caption=f'Ep {episode+1} | Step {step} | model.pt')
            frames.append(frame)
            step += 1

        print(f"Episode {episode + 1} 渲染完成（{len(frames)} 帧），准备保存 GIF...")
        save_path = os.path.join(gif_dir, f'eval_{episode + 1}.gif')
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=50, loop=0, disposal=2, optimize=False)
        print(f"GIF 已保存至: {save_path}")

    env.close()


if __name__ == '__main__':
    main()

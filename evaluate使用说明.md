# 固定镜头与 simple_tag 测评

在项目目录运行（python 应指向已安装依赖的环境）：

```powershell
Set-Location 'C:\Users\WEI\Desktop\code\maddpg-gpu-update'
conda activate dlrl
python evaluate.py simple_tag_v3 1 --device auto --episode-num 100 --episode-length 500 --boundary-mode bounce
python evaluate.py simple_tag_v3 1 --device auto --episode-num 100 --episode-length 500 --boundary-mode punish --boundary-penalty 100
```

其中 1 是模型结果目录编号，对应 results\simple_tag_v3\1\model.pt。默认场地为 [-1.5, 1.5] × [-1.5, 1.5]，可用 --boundary-range 修改。圆形物体按外缘触边，整个物体都限制在场地内。

simple_tag 可用 `--num-obstacles 0` 移除全部障碍物（真实移除碰撞与障碍物观测，不仅是隐藏图形），也支持 `--num_obstacles 0`。默认保留原有的 2 个障碍物。新训练会将障碍物数量写入 run_config.json，评估省略此参数时自动继承，旧配置未记录时默认 2。该参数只作用于 simple_tag。

移除障碍物会缩短观测向量，有障碍物训练的旧模型不能直接加载到无障碍物环境，需重新训练。示例：

```powershell
python main.py simple_tag_v3 --num-obstacles 0 --boundary-mode punish --boundary-penalty 100
python evaluate.py simple_tag_v3 N --num-obstacles 0 --episode-num 10 --episode-length 100
```

将 N 替换为这次无障碍物训练生成的结果目录编号；评估时也可省略 --num-obstacles，自动继承该模型的设置。普通动画、粒子轨迹动画与三项指标均支持无障碍物环境。

bounce：触边时截回边缘并反转朝外的法向速度分量，切向速度保持不变，不扣触边分。punish：截回边缘并清零朝外的法向速度，每个触边智能体每步扣分（默认 100），角落仅扣一次。用 --boundary-penalty 修改扣分值。速度仍受原环境阻尼和动作影响。离开边界即停止扣分。

训练与评估均支持 --boundary-mode、--boundary-range、--boundary-penalty，也支持下划线形式。省略评估边界参数时优先继承模型目录 run_config.json 中的训练设置；没有配置文件时使用 bounce / 1.5 / 100.0。已有配置中的旧扣分值会保留，需要覆盖时显式加 --boundary-penalty 100。旧模型可加载，建议重新训练以适应物理边界。

每次评估自动创建新的 evaluations\eval_001、eval_002 等目录，控制台显示保存位置：

- gif\episode_001.gif 等回合动画；
- trajectory_gif\episode_001_trajectory.gif 等粒子轨迹动画（默认额外保存）；
- evaluation_rewards.csv、evaluation_rewards.png：本次评估奖励及曲线；
- evaluation_metrics.csv：simple_tag 逐回合捕获成功状态、步数、逃逸数、各逃跑者首次捕获步数；
- evaluation_summary.json：三项指标汇总及本次边界设置；
- maddpg.log：本次评估日志。

普通 GIF 和轨迹 GIF 右下角默认显示即时积分面板：每个智能体的 total 是本回合截至当前帧的累计 reward，step 是当步 reward（含捕获奖励和触边惩罚），Sum 是所有智能体的合计。每回合从 0 开始，数值在环境完整执行一步后更新；累计分数与 evaluation_rewards.csv 一致。面板覆盖在画面右下角，不改变场地坐标或镜头比例。

simple_tag 中，物体外缘触碰场地边界时当帧显示为黑色；adversary 与 agent 按原环境捕获碰撞规则接触时，接触双方均显示为黑色。持续接触期间保持黑色，分离后恢复原角色颜色；同角色之间碰撞、障碍物接触不触发此项变色。bounce 和 punish 均显示触边变色，bounce 不会因此额外扣分。变色仅影响绘图，不修改仿真颜色、奖励或捕获统计。

粒子轨迹 GIF 在每一帧累积显示各智能体从本回合起点到当前步的运动路径，不显示未来位置。每个智能体使用独立轨迹颜色，左上方图例对应智能体名称，空心圆标记起点；粒子按上述规则在原角色颜色与接触黑色之间变化，轨迹本身保持独立颜色。固定障碍物不绘制运动路径。每回合轨迹独立，镜头、边界、帧率和分辨率与普通动画相同。

正常运行即可同时生成普通 GIF 和轨迹 GIF，例如 `python evaluate.py simple_tag_v3 1 --episode-num 10 --episode-length 100`。可显式加 `--trajectory-gif` 开启，或加 `--no-trajectory-gif` 关闭轨迹动画。双份动画会增加生成时间和内存占用，长回合可降低 --size。

捕获沿用原环境碰撞判定：中心距离小于双方半径之和。在每个完整环境步结束后检查，从第 1 步记录首次捕获。整局未被捕获才算逃逸；捕获后脱离不会恢复逃逸成功。回合仍运行至 episode-length，不因捕获提前结束。

三项指标：

- adversary_capture_success_rate：捕获成功回合数 / 总回合数；
- average_capture_steps：成功回合首次完成捕获的平均步数，无成功回合时为 JSON null，控制台显示 N/A；
- agent_escape_success_rate：从未被捕获的逃跑者数 / 所有回合逃跑者总数。

默认只有一个逃跑者，两种成功率互补。Python API 配置多个逃跑者时，团队成功要求该回合每个逃跑者都被捕获过，成功步数取最后一个首次被捕获的步数。文件中成功率用 0～1，控制台显示百分比。

镜头中心固定为 (0, 0)，--camera-range 2 显示横纵坐标 -2 到 2，坐标和半径采用相同比例。simple_tag 的相机范围必须不小于场地半宽。--size 700 控制 GIF 分辨率，--fps 25 控制播放速度，--seed 0 控制随机种子。--device auto 优先使用 GPU，也可指定 cpu 或 cuda。选动作保留原有 Gumbel-Softmax 随机采样逻辑。

其他环境仍可正常评估，例如：

```powershell
python evaluate.py simple_adversary_v3 3 --device auto --camera-range 2
```

三项捕获指标仅适用于 simple_tag，不会给其他环境输出误导的捕获统计。evaluate.py 不改动原模型、训练奖励或原训练日志。旧入口 evaluate_backup.py 也支持边界参数和三项指标，但指标文件写入模型目录，再次运行会更新同名指标文件。

本项目使用 Anaconda 的 dlrl 环境，解释器为 `C:\Users\WEI\.conda\envs\dlrl\python.exe`；编辑器默认解释器已指向此路径。已在该环境补装 NumPy 2.5.5，并将 PettingZoo 从 1.24.3 更新为 1.26.1，以提供 MPE2 所需的 AgentSelector。已直接通过边界/指标测试与 CUDA 模型评估，无需临时依赖路径。该环境为 Python 3.15，PettingZoo 1.26.1 的包元数据仅声明支持至 Python 3.14，此次安装跳过该版本限制；已验证本项目使用的接口，未验证其全部功能。

若终端尚未初始化 conda，也可直接指定解释器，例如：

```powershell
& 'C:\Users\WEI\.conda\envs\dlrl\python.exe' .\evaluate.py simple_tag_v3 1 --boundary-mode punish --boundary-penalty 100
```

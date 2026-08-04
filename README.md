# G1 数学开冰箱动作与 Isaac 3D 验证

这里不使用 CoorDex policy、checkpoint 或动作库。脚本直接读取 G1 Wuji 的 URDF，用连续逆运动学优化生成“移动根部/支撑路径、胸廓合理转身、右手保持整手 power grasp”的全身关键姿态。

求解变量包括浮动骨盆、双腿、腰和双臂；约束包括移动支撑路径、右掌闭链、圆柱轴向冗余、躯干转身、肩肘腕几何、关节限位和相邻姿态连续。随后将解对齐到 CoorDex Isaac 场景的真实冰箱铰链、门和把手位置，并使用服务器已有的 Isaac Sim 5.1 环境完成 RTX 3D 渲染。

## 生成结果

- `outputs/g1_fridge_action_keyframes.png`：0°、15°、30°、45°、60° 五个动作关键帧；
- `outputs/g1_fridge_action_overlay.png`：五个姿态叠加后的动作趋势；
- `outputs/g1_fridge_action.gif`：关键姿态往返播放；
- `outputs/frame_00.png` 至 `frame_04.png`：单独关键帧；
- `outputs/g1_fridge_action.npz`：完整优化变量、关节名称、门角和误差；
- `outputs/g1_fridge_action.csv`：便于查看的关节角表；
- `outputs/scene_reference.json`：从 Isaac 导出的机器人、冰箱和把手世界坐标；
- `outputs/isaac_aligned_action.npz`：按真实 Isaac 把手圆弧重新求解的动作；
- `outputs/g1_fridge_isaac_keyframes.png`：真实 Isaac Sim RTX 五关键帧；
- `outputs/g1_fridge_isaac.gif`：真实 Isaac Sim RTX 动图；
- `outputs/g1_fridge_isaac.mp4`：真实 Isaac Sim RTX 视频；
- `outputs/g1_fridge_isaac_static_side_keyframes.png`：启动时固定创建的无遮挡侧视关键帧；
- `outputs/g1_fridge_isaac_static_side.gif`：无遮挡侧视 25 帧动图；
- `outputs/g1_fridge_isaac_static_side.mp4`：无遮挡侧视视频；
- `outputs/isaac_balanced_grasp_action.npz`：第二版平衡与抓握轨迹；
- `outputs/g1_fridge_balanced_grasp_keyframes.png`：第二版全身关键帧；
- `outputs/g1_fridge_balanced_grasp.gif` / `.mp4`：第二版全身动作；
- `outputs/g1_fridge_balanced_grasp_closeup_keyframes.png`：第二版手部近景；
- `outputs/g1_fridge_balanced_grasp_closeup.gif` / `.mp4`：第二版手部近景动作；
- `RESULT_BALANCED_GRASP.md`：第二版目标、指标、改进幅度和局限；
- `outputs/g1_fridge_static_power_grasp_front.png`：真实把手网格约束后的正面静态握持；
- `outputs/g1_fridge_static_power_grasp_top.png`：同一姿态的俯视复核；
- `outputs/g1_fridge_static_power_grasp_two_views.png`：正面与俯视并排图；
- `outputs/static_power_grasp_final_validation.json`：逐掌面、逐指节全网格测试；
- `RESULT_STATIC_POWER_GRASP.md`：整手握持数学模型、失败根因、指标和复现方式；
- `outputs/isaac_wholebody_power_grasp_action.npz`：最终 0–60°、121 个闭链 IK 关键帧；
- `outputs/wholebody_power_grasp_strict_validation.json`：241 帧、36 项全身与整手严格测试；
- `outputs/wholebody_power_grasp_validation_{15,30,45}deg.json`：不同终止角的同套测试；
- `outputs/g1_fridge_wholebody_coordinated_full_keyframes.png` / `_full_10s.mp4`：最终全身 Isaac 结果；
- `outputs/g1_fridge_wholebody_coordinated_hand_keyframes.png` / `_hand_10s.mp4`：最终正侧手部结果；
- `outputs/g1_fridge_wholebody_coordinated_opposite_keyframes.png` / `_opposite_10s.mp4`：最终反侧复核；
- `outputs/g1_fridge_wholebody_coordinated_final_opposite.png`：60° 末帧抓持；
- `RESULT_WHOLEBODY_POWER_GRASP.md`：全身 + 整手闭链模型、严格测试和最终结论；
- `RESULT.md`：本轮模型、结果和局限。

## 运行

```bash
cd /Users/jason/Projects/msj-robotics/msj-loco-manip/CoorDex/fridge_motion_optimization_2026-08-05
python3 scripts/generate_action_keyframes.py
```

只需要本机现有的 NumPy、SciPy 和 Matplotlib。机器人几何来自：

```text
../source_repo/source/coordex/coordex/assets/g1_wuji/g1_wuji.urdf
```

当前图是由实际求解的 URDF 关节姿态渲染得到，不是概念图，也不是生成式图片。它仍是 kinematic prototype：还没有加入关节力矩、门阻力、足底摩擦锥和碰撞约束。

## 服务器 Isaac 环境

服务器工作目录为 `/sdb/mashijian/coordex/CoorDex`。旧的 `coordex` 环境是 Isaac Sim 5.0、Isaac Lab 2.2 和较旧 Warp，在驱动 580.142 上触发 `cuDeviceGetUuid` / CUDA error 36，随后 GPU PhysX 回退或卡住。

Doorman 已有环境位于 `/sdb/mashijian/coordex/envs/doorman`，对应 Isaac Sim 5.1.0、Isaac Lab 2.3.0、Warp 1.10.1、PyTorch 2.7.0+cu128。使用该环境后，Isaac 能识别 8 张 RTX 4090，当前渲染使用 `cuda:0`，没有再出现 CUDA error 36。

关键环境变量如下：

```bash
export LD_LIBRARY_PATH=/sdb/mashijian/coordex/envs/doorman/lib:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/sdb/mashijian/coordex/doorman/IsaacLab-2.3.0/source/isaaclab:/sdb/mashijian/coordex/CoorDex/source/coordex
```

服务器本轮中间文件保存在 `/sdb/mashijian/coordex/CoorDex/math_fridge_render`。

## 静态侧视相机方案

最终采用启动时静态相机方案：在创建环境之前，根据 eye `(-1.2, 2.1, 2.1)` 和 target `(1.0, 0.0, 0.85)` 计算 OpenGL 相机四元数，并将位置和旋转写入 `CameraCfg.OffsetCfg`。Camera prim 和 RTX render product 因而随场景一次性创建，不再在运行时移动或新建。

每个目标姿态写入后必须调用一次 `sim.step(render=True)`，否则 RTX annotator 会反复返回首帧缓存。最终版本已验证首尾帧约 19.8% 像素发生明显变化，画面中 G1、冰箱和门的 0° 到 60° 开启过程均完整可见。

服务器复现参数：

```bash
python math_fridge_render/render_isaac_keyframes.py \
  --trajectory math_fridge_render/isaac_aligned_action.npz \
  --scene-reference math_fridge_render/scene_reference.json \
  --output-dir math_fridge_render/static_tick_final \
  --interpolation-frames 25 \
  --camera-eye -1.2 2.1 2.1 \
  --camera-target 1.0 0.0 0.85 \
  --enable_cameras
```

动态移动 IsaacLab Camera 或运行时创建 Replicator render product 的诊断过程仍记录在 `RESULT.md`；错误诊断帧未放入 `outputs/`。

# 数学模型 V3：手—门闭链下的 DS→SS→DS 迈步规划

V2 只能回答“固定双脚何时不再合理”。V3 进一步生成一个可由 whole-body motion tracker（例如 SONIC）跟踪的任务空间参考：接触模式、迈步时机、摆动脚、落脚点、COM、骨盆和右腕闭链目标。

## 1. 接触模式

每个候选只允许以下结构：

```text
DS -> LS -> DS  （右脚摆动）
DS -> RS -> DS  （左脚摆动）
```

`DS` 为双支撑，`LS/RS` 为左/右单支撑。支撑脚在世界系严格固定；摆动脚在单支撑区间内用 minimum-jerk 水平轨迹和零端点导数的抬脚曲线移动。该定义从结构上排除了“两只脚贴地一起漂移”。

## 2. 手—门闭链

门角 `theta(t)` 决定把手与右腕目标：

```text
T_wrist*(t) = T_door(theta(t)) T_grasp
```

迈步规划器不改变右腕闭链目标，只通过骨盆、COM 和脚步重新布置机器人的支撑基座，从而在门角较大时恢复肩臂余量。

## 3. Centroidal 可行性

每个节点独立求解足底四角接触力：

```text
m c_ddot = sum(f_i) + f_hand + m g
L_dot     = sum((p_i-c) x f_i) + (p_hand-c) x f_hand
```

并约束：

```text
f_z >= 0
|f_x|, |f_y| <= mu/sqrt(2) f_z
COP inside active support polygon
```

单支撑节点只允许支撑脚产生接触力，摆动脚接触力严格为零。

## 4. 候选变量与选择

候选变量包括：

- 摆动脚（左或右）；
- 离地节点；
- 落地节点；
- 落脚位置 `(x, y)`；
- 落脚 yaw。

先用手臂未来可达性、步长和 COM 平滑性对候选排序，再对最优若干候选执行完整 centroidal 接触力审计。选择满足以下条件且代价最小的候选：

- 支撑脚零漂移；
- 摆动脚达到最小净空；
- 全部 DS/SS 节点接触力可行；
- 摩擦与 COP 有非负余量；
- 肩到腕距离低于给定上限；
- COM 加速度低于上限。

## 5. 输出给 motion tracker

```text
outputs/sonic_closed_chain_step_reference.npz
```

包含：

- `support_mode`；
- 左右脚目标位置/yaw；
- COM 目标与加速度；
- 骨盆目标位置/yaw；
- 右腕闭链目标；
- 门对机器人的手部反力；
- 每帧接触可行性、COP 和摩擦余量。

SONIC 被视为下游 whole-body motion tracker。V3 负责“参考运动与接触模式是什么”，而不是重新训练或替代底层跟踪器。

## 6. 运行

```bash
python scripts/run_math_model_v3.py \
  --input outputs/isaac_full_horizon_fixed_support_action.npz \
  --audit outputs/full_horizon_quasistatic_audit.json \
  --duration 8 \
  --fail-on-infeasible
```

也可以分开运行：

```bash
python scripts/plan_closed_chain_step.py
python scripts/validate_closed_chain_step.py --fail-on-error
```

## 7. 验证范围

CI 中的合成测试验证：

1. DS→SS→DS 模式连续且唯一；
2. 支撑脚无漂移；
3. 摆动脚有足够离地净空并精确落脚；
4. 正常载荷下单支撑 centroidal 平衡可行；
5. 迈步后后半程肩腕距离得到改善；
6. 过大的门反力会被判为接触不可行，而不是靠脚滑动通过。

完整 G1/Isaac 场景验证仍需在仓库已有服务器环境中运行上述 V3 命令，并将生成的 task-space reference 交给 SONIC 跟踪。

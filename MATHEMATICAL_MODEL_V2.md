# 数学模型 V2：全时域准静态优化

旧求解器逐门角运行局部 IK，并允许根部与双脚沿人为支撑路径一起移动。门角增大后，误差容易集中到肩、肘和腕；脚目标又跟随机器人移动，因此验证器无法区分真实迈步与整机漂移。

V2 将问题拆成两个独立、可审计的数学层：

1. **全时域固定支撑闭链优化**：整段门角状态一次性联合求解；双脚在世界系固定，手掌保持把手闭链，同时优化 COM、关节限位余量、肩肘几何以及速度、加速度和 jerk。
2. **独立准静态力学审计**：由门铰链圆、门阻力矩和把手杠杆计算手部切向反力；求解双脚四角接触力，使 Newton–Euler 平衡、单边接触与摩擦锥同时成立；检查 COP、右臂 Jacobian 奇异性、关节余量和脚漂移。

若固定双脚在某一门角后失效，输出会明确报告“需要改变接触模式”，而不是移动脚目标来伪造可达性。

## 模型

给定门阻力矩 `tau_d`，把手切向力满足：

```text
|f_hand| = tau_d / handle_radius
```

固定双脚四角接触力满足：

```text
sum(f_i) + f_hand + m g = 0
sum((p_i-p_com) x f_i) + (p_hand-p_com) x f_hand = 0
f_z >= 0
sqrt(f_x^2 + f_y^2) <= mu f_z
```

全时域变量为：

```text
X = [x_0, ..., x_N]
x_k = [root position, root orientation, body joints]
```

所有节点联合最小化手掌闭链、固定脚、COM、关节余量、肩臂几何和全时域平滑项。与旧方法不同，早期姿态会受到后期可达性的影响。

## 运行

```bash
python -m pip install -r requirements-math.txt
python scripts/run_math_model_v2.py --nodes 13 --duration 10
```

若自动化流程需要在“固定支撑不再合理”时失败：

```bash
python scripts/run_math_model_v2.py --fail-on-step-required
```

输出：

- `outputs/isaac_full_horizon_fixed_support_action.npz`
- `outputs/full_horizon_quasistatic_audit.json`

## 当前边界

V2 是全时域准静态固定支撑模型，不伪造真实迈步。下一阶段需要显式加入双支撑、左支撑、右支撑三种接触模式，摆动脚离地/落脚约束和 centroidal dynamics。

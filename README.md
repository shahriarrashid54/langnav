# LangNav: Natural Language Robot Navigation

Autonomous mobile robot navigation using Vision-Language Models + Reinforcement Learning. Maps natural language instructions to robot movements via YOLO object detection + CLIP vision-language understanding.

## Architecture

```
"Go to the red box"
     ↓
[CLIP] → Parse intent + locate target
     ↓
[YOLOv11] → Real-time object tracking
     ↓
[PPO Agent] → Learned navigation policy (trained in sim)
     ↓
[ROS2 Nav2] → Execute movement commands
```

## Stack

- **Robot Framework:** ROS2 Humble
- **Simulation:** Gazebo Fortress
- **Vision:** YOLOv11 (Ultralytics) + CLIP
- **RL Training:** Stable-Baselines3 (PPO)
- **Deep Learning:** PyTorch

## Project Structure

```
langnav/
├── ros2_ws/              # ROS2 workspace
│   └── src/langnav_robot/
│       ├── langnav_core/     # Core orchestration node
│       ├── langnav_vision/   # YOLO + CLIP pipeline
│       ├── langnav_rl/       # RL training environment
│       └── langnav_sim/      # Gazebo simulation
├── configs/              # YAML configs (models, training)
├── scripts/              # Training + setup scripts
├── docker/               # Dockerfile for reproducibility
└── tests/                # Unit + integration tests
```

## Quick Start

```bash
# Build
cd langnav
./scripts/setup_env.sh

# Train RL agent
python scripts/train_model.py --config configs/ppo_nav.yaml

# Run in simulation
ros2 launch langnav_sim gazebo.launch.py
```

## Roadmap

- [ ] YOLO object detection pipeline
- [ ] CLIP vision-language integration
- [ ] Gazebo simulation environment
- [ ] PPO RL training loop
- [ ] ROS2 navigation node
- [ ] Full end-to-end demo
- [ ] W&B metrics tracking
- [ ] Docker containerization

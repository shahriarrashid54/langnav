.PHONY: help install test train demo lint format

MODEL ?= checkpoints/langnav_ppo_v1_final

help:
	@echo "LangNav Build Targets"
	@echo "===================="
	@echo "install       - Install dependencies"
	@echo "test          - Run tests"
	@echo "train         - Train PPO model (simple backend)"
	@echo "train-wandb   - Train with W&B logging"
	@echo "eval          - Evaluate checkpoint (MODEL=path)"
	@echo "demo          - Record demo GIFs (MODEL=path)"
	@echo "lint          - Run linter"
	@echo "format        - Format code"
	@echo "docker-build  - Build Docker image"
	@echo "docker-run    - Run Docker container"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=langnav

train:
	python scripts/train_model.py --config configs/ppo_nav.yaml --total-timesteps 1000000

train-wandb:
	python scripts/train_model.py --config configs/ppo_nav.yaml --wandb

eval:
	python scripts/train_model.py --eval-only $(MODEL) --config configs/ppo_nav.yaml

demo:
	python scripts/record_demo.py --model $(MODEL) --output-dir demo --n-episodes 20 --fps 15

lint:
	black --check langnav tests scripts
	flake8 langnav tests scripts

format:
	black langnav tests scripts

docker-build:
	docker build -f docker/Dockerfile -t langnav:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

sim-build:
	docker-compose -f docker/docker-compose.sim.yml build

sim-run:
	docker-compose -f docker/docker-compose.sim.yml up

sim-gazebo:
	docker exec -it langnav_sim bash -c \
	  "source /opt/ros/humble/setup.bash && \
	   ros2 launch langnav_sim gazebo.launch.py rviz:=true"

sim-shell:
	docker exec -it langnav_sim bash

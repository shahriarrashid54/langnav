.PHONY: help install test train lint format

help:
	@echo "LangNav Build Targets"
	@echo "===================="
	@echo "install       - Install dependencies"
	@echo "test          - Run tests"
	@echo "train         - Train PPO model"
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

lint:
	black --check langnav tests scripts
	flake8 langnav tests scripts

format:
	black langnav tests scripts

docker-build:
	docker build -f docker/Dockerfile -t langnav:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

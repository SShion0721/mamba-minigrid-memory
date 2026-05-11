"""Convenience entrypoint for the no-memory PPO-MLP baseline."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_mamba_ppo import main


if __name__ == "__main__":
    main(default_model="mlp")

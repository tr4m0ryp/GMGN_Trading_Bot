#!/usr/bin/env python3
"""
Model inference server for AI Trader.

Reads observation from stdin, outputs prediction to stdout.
Designed for fast subprocess calls from C.

Usage:
    echo "0.1,0.2,..." | python inference.py --model model.zip

Author: Trading Team
Date: 2025-12-24
"""

import sys
import os
import json
import time
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'ai_model', 'src'))

def load_model(model_path: str):
    """Load stable-baselines3 PPO model."""
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        return model
    except ImportError:
        print(json.dumps({
            "success": False,
            "error": "stable-baselines3 not installed. Run: pip install stable-baselines3"
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Failed to load model: {str(e)}"
        }))
        sys.exit(1)


def predict(model, observation: np.ndarray) -> dict:
    """Run inference and return result."""
    start_time = time.time()
    
    try:
        # Get action and probabilities
        action, _states = model.predict(observation, deterministic=True)
        
        # Get action probabilities
        obs_tensor = model.policy.obs_to_tensor(observation)[0]
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()[0]
        
        inference_time_us = int((time.time() - start_time) * 1_000_000)
        
        return {
            "success": True,
            "action": int(action),
            "confidence": float(probs[action]),
            "probabilities": [float(p) for p in probs],
            "inference_time_us": inference_time_us
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action": 0,
            "confidence": 0.0,
            "probabilities": [1.0, 0.0, 0.0],
            "inference_time_us": 0
        }


def main():
    parser = argparse.ArgumentParser(description='Model inference server')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--batch', action='store_true', help='Batch mode (read multiple lines)')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    if args.batch:
        # Batch mode: read multiple observations
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                obs = np.array([float(x) for x in line.split(',')], dtype=np.float32)
                result = predict(model, obs)
                print(json.dumps(result), flush=True)
            except Exception as e:
                print(json.dumps({"success": False, "error": str(e)}), flush=True)
    else:
        # Single observation mode
        line = sys.stdin.read().strip()
        if not line:
            print(json.dumps({"success": False, "error": "No observation provided"}))
            return
        
        try:
            obs = np.array([float(x) for x in line.split(',')], dtype=np.float32)
            result = predict(model, obs)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))


if __name__ == '__main__':
    main()

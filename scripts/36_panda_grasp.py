"""Day 11 无界面运行：PYTHONPATH=. python scripts/36_panda_grasp.py --headless。

macOS 可视化运行：PYTHONPATH=. mjpython scripts/36_panda_grasp.py
"""

import argparse
import json

from robotics.grasp_experiment import load_cube_scene, run_experiment


def main():
    """共用同一实验入口，支持无界面验证或 macOS Viewer 可视化。"""
    parser = argparse.ArgumentParser(description="Day 11 physical cube grasp/lift/release")
    parser.add_argument("--headless", action="store_true", help="Run without importing Viewer.")
    args = parser.parse_args()
    model, data = load_cube_scene()
    if args.headless:
        result = run_experiment(model, data)
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.4, 0., 0.30]
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            result = run_experiment(model, data, viewer=viewer)
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit("Day 11 trial failed; inspect contact and pose metrics above.")


if __name__ == "__main__":
    main()

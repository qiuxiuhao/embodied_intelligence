"""Mac 用 mjpython 加载已训练模型；与无界面评估共享相同推理代码。"""

import sys

from training.evaluate_ppo_reach import main


if __name__ == "__main__":
    if "--human" not in sys.argv:
        sys.argv.append("--human")
    main()

"""按 geom 标识读取 MuJoCo 求解后的接触信息；力的单位均为 N。"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CubeContact:
    """一个有效接触约束：位置使用世界坐标（m），单位法向指向 Cube 内部。"""

    geom1: int
    geom2: int
    other_body: int
    position: np.ndarray
    normal_on_cube: np.ndarray
    normal_force_N: float


@dataclass(frozen=True)
class ContactSnapshot:
    """瞬时接触证据；抬升成功还需满足持续时间要求。"""

    cube_contacts: tuple
    finger_contact: bool
    grasp_candidate: bool
    cube_table: bool
    cube_floor: bool
    finger_table: bool
    left_force_N: float
    right_force_N: float


class PandaContactDetector:
    """通过所属 body 的 ID 匹配官方模型中未命名的手指碰撞 geom。"""

    def __init__(self, model, min_force_N=0.05):
        """要求求解后的法向力达到正阈值，排除仅进入接触裕量却尚未实际受力的接触对。"""
        if not np.isfinite(min_force_N) or min_force_N <= 0:
            raise ValueError("min_force_N must be finite and positive.")
        self.model = model
        self.min_force_N = min_force_N
        self.cube = int(model.geom("cube_geom").id)
        self.table = int(model.geom("table").id)
        self.floor = int(model.geom("floor").id)
        self.left = int(model.body("left_finger").id)
        self.right = int(model.body("right_finger").id)
        self.fingers = {self.left, self.right}

    def read(self, data):
        """在完整调用 mj_forward 后读取 data.contact 与 mj_contactForce。

        接触坐标系的第 0 行法向从 geom1 指向 geom2；当 Cube 是 geom1 时需反向，
        使左右手指的接触法向都表示世界坐标系下作用于 Cube 的受力方向。
        双侧接触必须分居闭合轴两侧，且法向相对；仅出现两个 geom 或 data.ncon > 0 不足以判定夹持。
        """
        records = []
        finger_table = False
        for i in range(data.ncon):
            c = data.contact[i]
            if c.efc_address < 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.cube not in (g1, g2) and self.table not in (g1, g2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            if force[0] < self.min_force_N:
                continue
            if self.table in (g1, g2):
                other = g2 if g1 == self.table else g1
                finger_table |= int(self.model.geom_bodyid[other]) in self.fingers
            if self.cube not in (g1, g2):
                continue
            other = g2 if g1 == self.cube else g1
            normal = c.frame[:3].copy() * (1 if g2 == self.cube else -1)
            records.append(CubeContact(g1, g2, int(self.model.geom_bodyid[other]),
                                       c.pos.copy(), normal, float(force[0])))
        left = [r for r in records if r.other_body == self.left]
        right = [r for r in records if r.other_body == self.right]
        axis = data.body("hand").xmat.reshape(3, 3)[:, 1]
        center = data.geom("cube_geom").xpos
        candidate = any(
            np.dot(l.position - center, axis) > 0.005
            and np.dot(r.position - center, axis) < -0.005
            and np.dot(l.normal_on_cube, axis) < -0.5
            and np.dot(r.normal_on_cube, axis) > 0.5
            and np.dot(l.normal_on_cube, r.normal_on_cube) < -0.5
            for l in left for r in right
        )
        has_geom = lambda g: any(g in (r.geom1, r.geom2) for r in records)
        return ContactSnapshot(tuple(records), bool(left or right), candidate,
                               has_geom(self.table), has_geom(self.floor), finger_table,
                               sum(r.normal_force_N for r in left),
                               sum(r.normal_force_N for r in right))

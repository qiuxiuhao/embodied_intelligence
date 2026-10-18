"""Panda 腱传动夹爪控制；长度单位 m，力单位 N，时间单位 s。"""

import mujoco
import numpy as np


class PandaGripperController:
    """将名义开度（两侧滑动关节位移之和）映射为 actuator8 的控制输入。

    开度是关节空间量，并非所有指垫表面间距的精确标定值。
    是否形成夹持由实际接触几何与受力决定。
    """

    def __init__(self, model):
        """校验模型实际的固定腱传动与仿射伺服配置，再推导开度到控制输入的映射。"""
        self.model = model
        self.actuator_id = int(model.actuator("actuator8").id)
        a = self.actuator_id
        self.joint_ids = np.array([model.joint(n).id for n in
                                   ("finger_joint1", "finger_joint2")])
        if (a != 7 or model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_TENDON
                or model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE
                or model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
                or not model.actuator_ctrllimited[a]):
            raise ValueError("Expected official Panda eighth fixed-tendon affine servo.")
        self.tendon_id = int(model.actuator_trnid[a, 0])
        adr = int(model.tendon_adr[self.tendon_id])
        indices = slice(adr, adr + int(model.tendon_num[self.tendon_id]))
        if (not np.all(model.wrap_type[indices] == mujoco.mjtWrap.mjWRAP_JOINT)
                or not np.array_equal(model.wrap_objid[indices], self.joint_ids)
                or not np.allclose(model.wrap_prm[indices], [0.5, 0.5])
                or not np.all(model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError("Expected split = 0.5*finger_joint1 + 0.5*finger_joint2.")
        self.ctrl_range = model.actuator_ctrlrange[a].copy()
        gain = float(model.actuator_gainprm[a, 0])
        bias0, bias1 = model.actuator_biasprm[a, :2]
        gear = float(model.actuator_gear[a, 0])
        if gain <= 0 or bias1 >= 0 or gear <= 0:
            raise ValueError("Unsupported gripper servo coefficients.")
        # 静止时满足：gain*u + bias0 + bias1*(gear*tendon_length) = 0。
        self.ctrl_per_meter = float(-bias1 * gear * 0.5 / gain)
        self.ctrl_offset = float(-bias0 / gain)
        joint_range = model.jnt_range[self.joint_ids].sum(axis=0)
        servo_range = (self.ctrl_range - self.ctrl_offset) / self.ctrl_per_meter
        self.width_range = np.array([max(joint_range[0], servo_range[0]),
                                     min(joint_range[1], servo_range[1])])
        if self.width_range[0] >= self.width_range[1]:
            raise ValueError("Empty gripper opening range.")

    def command(self, data, opening_m):
        """发送限幅后的开度目标（m），返回实际写入的原生 ctrl 控制量。"""
        if not np.isfinite(opening_m):
            raise ValueError("opening_m must be finite.")
        width = np.clip(opening_m, *self.width_range)
        ctrl = float(np.clip(self.ctrl_offset + self.ctrl_per_meter * width,
                             *self.ctrl_range))
        data.ctrl[self.actuator_id] = ctrl
        return ctrl

    def feedback(self, data):
        """分别读取两侧滑动关节的实测值，不以命令值或等式约束近似代替反馈。"""
        q = data.qpos[self.model.jnt_qposadr[self.joint_ids]].copy()
        v = data.qvel[self.model.jnt_dofadr[self.joint_ids]].copy()
        return {"finger_position_m": q, "finger_velocity_m_s": v,
                "opening_m": float(q.sum()),
                "tendon_length_m": float(data.ten_length[self.tendon_id]),
                "actuator_force_N": float(data.actuator_force[self.actuator_id])}

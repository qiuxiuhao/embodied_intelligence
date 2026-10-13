from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "actuated_joint.xml"

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)

print("========== MODEL ==========")

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)


print("\n========== STATE ==========")

print("qpos:")
print(data.qpos)

print("qvel:")
print(data.qvel)


print("\n========== CONTROL ==========")

print("ctrl:")
print(data.ctrl)

print("ctrl shape:")
print(data.ctrl.shape)

motor = model.actuator("joint_motor")

motor_id = motor.id

print("\n========== ACTUATOR ==========")
print("motor_id:", motor_id)
print(
    "ctrl range:",
    model.actuator_ctrlrange[motor_id],
)
print(
    "gear:",
    model.actuator_gear[motor_id],
)
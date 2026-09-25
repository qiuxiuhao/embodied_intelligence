import mujoco

xml = """
<mujoco>
    <worldbody>
        <body pos ="0 0 1">
            <freejoint/>
                <geom
                    type = "sphere"
                    size = "0.1"
                    mass = "1"
            />
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)

data = mujoco.MjData(model)

print("===== MuJoCo Smoke Test =====")

print("Initial time:")
print(data.time)

print("Initial qpos:")
print(data.qpos)

print("Initial qvel:")
print(data.qvel)


for i in range(10):
    mujoco.mj_step(model, data)

print("\nAfter 10 simulation steps:")

print("time:")
print(data.time)

print("qpos:")
print(data.qpos)

print("qvel:")
print(data.qvel)
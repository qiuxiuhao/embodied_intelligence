# Day 3 - ACTUATOR 与 Control
## Joint VS ACtuator

 joint： 规定body如何运动
 actuator：对 joint / tendon / site 等参数主动控制作用

## Control

data.ctrl:mujoco actuator 的控制输入
nu: control vector 的维数

加入一个 scalar motor后：
nq = 1    
nv = 1
nu = 1

## Action 与 ctrl
action 是 agent 层概念
data.ctrl 是MuJoCo actuator 输入
简单环境中： action= ctrl
复杂机器人中可能：
Action ——> Controller / IK ——> Joint command ——> data.ctrl

## Open-loop
不读取当前state: time -> action -> robot

## closed-loop
state -> controllor -> action ->robot -> new state

## PD Controller

error = target -q
action = Kp * error - kd * qvel

P: 把系统拉向目标
D： 抑制速度和震荡

## Control Saturation
actuator 存在有限 ctrlrange

## Agent-Environment Loop
Observation -> Policy -> Action -> Environment -> Next Observation



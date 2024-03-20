import roboticstoolbox as rtb
import swift
import spatialgeometry as sg
import spatialmath as sm

class RobotViz():
    def __init__(self):
        self.env = swift.Swift()
        self.env.launch()
        self.robot = rtb.models.Panda()
        self.env.add(self.robot)
        self.object_pose = sg.Axes(0.1, pose = sm.SE3(1,1,1))
        self.env.add(self.object_pose)
        self.robot.grippers[0].q = [0.03, 0.03]
    
    def step(self, q):
        self.robot.q = q
        self.env.step()
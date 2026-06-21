class Simulator:

    def __init__(self, root_node, info):
        self.info = info
        self.root_node = root_node
        self.dt = self.info[5][0]["step"]
        self.gravity = self.info[5][0]["gravity"]
        self.friction_coefficient = self.info[5][0]["friction_coefficient"]
        self.alarmDistance = self.info[5][0]["contactDistance"]*2
        self.contactDistance = self.info[5][0]["contactDistance"]
        self.background_color = self.info[5][0]["background_color"]
        self.Create_Simulator()

    def Create_Simulator(self):
        for plugin in [
            'BeamAdapter',
            'Sofa.Component.AnimationLoop',
            'Sofa.Component.Collision.Detection.Algorithm',
            'Sofa.Component.Collision.Detection.Intersection',
            'Sofa.Component.Collision.Geometry',
            'Sofa.Component.Collision.Response.Contact',
            'Sofa.Component.Constraint.Lagrangian.Correction',
            'Sofa.Component.Constraint.Lagrangian.Solver',
            'Sofa.Component.Constraint.Projective',
            'Sofa.Component.Engine.Select',
            'Sofa.Component.IO.Mesh',
            'Sofa.Component.LinearSolver.Direct',
            'Sofa.Component.Mapping.Linear',
            'Sofa.Component.Mapping.NonLinear',
            'Sofa.Component.Mass',
            'Sofa.Component.MechanicalLoad',
            'Sofa.Component.ODESolver.Backward',
            'Sofa.Component.Setting',
            'Sofa.Component.SolidMechanics.FEM.Elastic',
            'Sofa.Component.SolidMechanics.Spring',
            'Sofa.Component.StateContainer',
            'Sofa.Component.Topology.Container.Constant',
            'Sofa.Component.Topology.Container.Dynamic',
            'Sofa.Component.Topology.Container.Grid',
            'Sofa.Component.Topology.Mapping',
            'Sofa.Component.Visual',
            'Sofa.GL.Component.Rendering2D',
            'Sofa.GL.Component.Rendering3D',
            'Sofa.GL.Component.Shader',
        ]:
            self.root_node.addObject('RequiredPlugin', name=plugin)
        self.root_node.findData('gravity').value = self.gravity
        self.root_node.findData('dt').value = self.dt
        self.root_node.addObject('VisualStyle')
        self.root_node.addObject('FreeMotionAnimationLoop')
        self.root_node.addObject('CollisionPipeline')
        self.root_node.addObject('BruteForceBroadPhase')
        self.root_node.addObject('BVHNarrowPhase')
        self.root_node.addObject('LocalMinDistance',
                           alarmDistance=self.alarmDistance, contactDistance=self.contactDistance,
                           angleCone=0.01)
        self.root_node.addObject('RuleBasedContactManager', responseParams="mu=" + str(self.friction_coefficient),
                           name='Response', response='FrictionContactConstraint')
        self.constraints = self.root_node.addObject('BlockGaussSeidelConstraintSolver', tolerance=1e-6, maxIterations=1000, computeConstraintForces=1)
        background_color = list(self.background_color)
        if len(background_color) == 3:
            background_color.append(1)
        self.root_node.addObject('BackgroundSetting', color=background_color)
        self.root_node.addObject('LightManager', name="lightManager",  listening="1")
        self.root_node.addObject('DirectionalLight', name="light1", color="0.7451 0.7451 0.7451", direction="0 -1 -1")
        self.root_node.addObject('DirectionalLight', name="light2", color="0.7451 0.7451 0.7451", direction="0 1 0")
        self.root_node.addObject('DirectionalLight', name="light3", color="0.7451 0.7451 0.7451", direction="0 0 1")
        if self.info[5][0].get('cam', 0) == 1 and self.info[2]:
            if self.info[2][0]['object'] != "capsule":
                raise ValueError('Imitation-learned control camera follows object="capsule" only.')
            self.cam = self.root_node.addObject("OglViewport", screenPosition=[0, -400], screenSize=[400, 400], cameraRigid=[0, 0, 0, 0, 0, 0, 1],  zNear="0.005", zFar="1", fovy="30", drawCamera=0)

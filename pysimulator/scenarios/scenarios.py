import pymht.utils.simulator as sim
import pymht.models.pv as pv_model
import pymht.models.ais as ais_model
from pymht.utils import classDefinitions
from pymht.utils.classDefinitions import SimTarget, SimTargetCartesian, SimTargetPolar
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import numpy as np
import copy


class ScenarioBase:

    def __init__(self):
        # Static scenario data
        self.staticSeed = 5447
        self.initTime = 0.
        self.radarPeriod = 60. / 24.  # 24 RPM radar / 48 RPM radar
        self.radarRange = 5500.0  # meters
        self.simulationTimeStep = self.radarPeriod / 4  # sec
        self.simTime = self.radarPeriod * 60  # sec
        self.nScans = int(self.simTime / self.radarPeriod)
        self.nSimulationSteps = int(self.simTime / self.simulationTimeStep)
        self.P_d_true = 0.8  # Probability of detection
        self.sigma_Q = pv_model.sigmaQ_true
        self.P_r = 0.9  # Probability of receive (AIS)
        self.model = pv_model
        self.p0 = np.array([100., -100.])  # own position
        # Expected number of measurements from new targets
        # per unit area of the measurement space
        self.lambda_nu = 2e-8
        self.lambda_local = 1

        assert self.simulationTimeStep <= self.radarPeriod
        assert self.simTime >= self.simulationTimeStep
        assert self.nScans >= 1

    def __str__(self):
        return str(vars(self).items())

    __repr__ = __str__

    def storeScenarioSettings(self, scenarioElement):
        scenariosettingsElement = ET.SubElement(scenarioElement, scenariosettingsTag)
        for k, v in vars(self).items():
            ET.SubElement(scenariosettingsElement, str(k)).text = str(v)


class Scenario(ScenarioBase):

    def __init__(self, name):
        ScenarioBase.__init__(self)
        self.name = name
        self.initialTargets = []

    def __getitem__(self, item):
        return self.initialTargets.__getitem__(item)

    def __iter__(self):
        return self.initialTargets.__iter__()

    def __len__(self):
        return self.initialTargets.__len__()

    def __str__(self):
        return "\n".join([str(e) for e in self.initialTargets])

    def __copy__(self):
        newScenario = Scenario(copy.copy(self.name))
        newScenario.initialTargets = [copy.copy(e) for e in self.initialTargets]
        return newScenario

    def append(self, newTarget):
        if not isinstance(newTarget,SimTarget):
            raise ValueError("Wrong input type. Must be SimTarget")
        self.initialTargets.append(newTarget)

    def addCartesian(self, state, **kwargs):
        default = {'probabilityOfReceive': self.P_r}
        self.initialTargets.append(SimTargetCartesian(state,
                                             self.initTime,
                                             self.P_d_true,
                                             self.sigma_Q,
                                             **{**default, **kwargs}))

    def addPolar(self, state, **kwargs):
        default = {'probabilityOfReceive': self.P_r, 'headingChangeMean': 0}
        self.initialTargets.append(SimTargetPolar(state,
                                                  self.initTime,
                                                  self.P_d_true,
                                                  self.sigma_Q,
                                                  **{**default, **kwargs}))

    def getSimList(self, simTime=None):
        if simTime is None:
            simTime = self.simTime
        sim.seed_simulator(self.staticSeed)
        return sim.simulateTargets(self.initialTargets,
                                   simTime,
                                   self.simulationTimeStep,
                                   pv_model)

    def getSimulatedScenario(self, seed, simList, lambda_phi, P_d, **kwargs):
        sim.seed_simulator(seed)

        scanList = sim.simulateScans(simList,
                                     self.radarPeriod,
                                     pv_model.C_RADAR,
                                     pv_model.R_RADAR(pv_model.sigmaR_RADAR_true),
                                     lambda_phi,
                                     self.radarRange,
                                     self.p0,
                                     P_d=P_d,
                                     lambda_local=self.lambda_local,
                                     **kwargs)

        aisList = sim.simulateAIS(simList,
                                  ais_model,
                                  self.radarPeriod,
                                  self.initTime,
                                  **kwargs)
        return scanList, aisList

# Scenario 0
scenario0 = Scenario("Scenario0")
scenario0.addPolar([-2000, 2100, 270, 10], headingChangeMean=0.9, timeOfLastAisMessage=-2)
scenario0.addCartesian([100, -2000, -2, 8], timeOfLastAisMessage=-1)
scenario0.addCartesian([-4000, 300, 12, -1], timeOfLastAisMessage=-5)
scenario0.addPolar([-4000, 0, 90, 12], headingChangeMean=-0.2)
scenario0.addCartesian([-4000, -200, 17, 1], timeOfLastAisMessage=-4)
scenario0.addCartesian([4000, -2000, 1, -8], timeOfLastAisMessage=-3)
scenario0.addCartesian([3000, 4000, 2, -8], timeOfLastAisMessage=-2)
scenario0.addCartesian([200, 5000, 10, -1], timeOfLastAisMessage=-1)
scenario0.addCartesian([-3500, -3500, 10, 5], timeOfLastAisMessage=-6)
scenario0.addCartesian([-4100, 3200, 17, -2], timeOfLastAisMessage=-10)
scenario0.addCartesian([3600, 3000, -10, 3], timeOfLastAisMessage=-7)
scenario0.addCartesian([5000, 1000, -7, -2], timeOfLastAisMessage=-5)
scenario0.addCartesian([2000, 100, -10, 8], timeOfLastAisMessage=-2)
scenario0.addCartesian([0, -5000, 10, 2], timeOfLastAisMessage=-4)
scenario0.addCartesian([-400, 300, 17, 0], timeOfLastAisMessage=-8)
scenario0.addCartesian([0, 2000, 15, 15], timeOfLastAisMessage=-9)

# Scenario 1
scenario1 = copy.copy(scenario0)
scenario1.name = "Scenario1"
scenario1.initialTargets[0].mmsi = 257114400
scenario1.initialTargets[0].aisClass = 'B'
scenario1.initialTargets[2].mmsi = 257114402
scenario1.initialTargets[2].aisClass = 'B'
scenario1.initialTargets[4].mmsi = 257114404
scenario1.initialTargets[4].aisClass = 'B'
scenario1.initialTargets[6].mmsi = 257114406
scenario1.initialTargets[6].aisClass = 'B'
scenario1.initialTargets[8].mmsi = 257114408
scenario1.initialTargets[8].aisClass = 'B'
scenario1.initialTargets[10].mmsi = 257114410
scenario1.initialTargets[10].aisClass = 'B'
scenario1.initialTargets[12].mmsi = 257114412
scenario1.initialTargets[12].aisClass = 'B'
scenario1.initialTargets[14].mmsi = 257114414
scenario1.initialTargets[14].aisClass = 'B'

# Scenario 2
scenario2 = copy.copy(scenario1)
scenario2.name = "Scenario2"
scenario2.initialTargets[0].aisClass = 'A'
scenario2.initialTargets[1].aisClass = 'A'
scenario2.initialTargets[2].aisClass = 'A'
scenario2.initialTargets[3].aisClass = 'A'
scenario2.initialTargets[4].aisClass = 'A'
scenario2.initialTargets[5].aisClass = 'A'
scenario2.initialTargets[6].aisClass = 'A'
scenario2.initialTargets[7].aisClass = 'A'
scenario2.initialTargets[8].aisClass = 'A'
scenario2.initialTargets[9].aisClass = 'A'
scenario2.initialTargets[10].aisClass = 'A'
scenario2.initialTargets[11].aisClass = 'A'
scenario2.initialTargets[12].aisClass = 'A'
scenario2.initialTargets[13].aisClass = 'A'
scenario2.initialTargets[14].aisClass = 'A'
scenario2.initialTargets[15].aisClass = 'A'

# Scenario 3
scenario3 = copy.copy(scenario0)
scenario3.name = "Scenario3"
scenario3.initialTargets[0].mmsi = 257114400
scenario3.initialTargets[0].aisClass = 'B'
scenario3.initialTargets[1].mmsi = 257114401
scenario3.initialTargets[1].aisClass = 'B'
scenario3.initialTargets[2].mmsi = 257114402
scenario3.initialTargets[2].aisClass = 'B'
scenario3.initialTargets[3].mmsi = 257114403
scenario3.initialTargets[3].aisClass = 'B'
scenario3.initialTargets[4].mmsi = 257114404
scenario3.initialTargets[4].aisClass = 'B'
scenario3.initialTargets[5].mmsi = 257114405
scenario3.initialTargets[5].aisClass = 'B'
scenario3.initialTargets[6].mmsi = 257114406
scenario3.initialTargets[6].aisClass = 'B'
scenario3.initialTargets[7].mmsi = 257114407
scenario3.initialTargets[7].aisClass = 'B'
scenario3.initialTargets[8].mmsi = 257114408
scenario3.initialTargets[8].aisClass = 'B'
scenario3.initialTargets[9].mmsi = 257114409
scenario3.initialTargets[9].aisClass = 'B'
scenario3.initialTargets[10].mmsi = 257114410
scenario3.initialTargets[10].aisClass = 'B'
scenario3.initialTargets[11].mmsi = 257114411
scenario3.initialTargets[11].aisClass = 'B'
scenario3.initialTargets[12].mmsi = 257114412
scenario3.initialTargets[12].aisClass = 'B'
scenario3.initialTargets[13].mmsi = 257114413
scenario3.initialTargets[13].aisClass = 'B'
scenario3.initialTargets[14].mmsi = 257114414
scenario3.initialTargets[14].aisClass = 'B'
scenario3.initialTargets[15].mmsi = 257114415
scenario3.initialTargets[15].aisClass = 'B'

# Scenario 4
scenario4 = copy.copy(scenario3)
scenario4.name = "Scenario4"
scenario4.initialTargets[0].aisClass = 'A'
scenario4.initialTargets[1].aisClass = 'A'
scenario4.initialTargets[2].aisClass = 'A'
scenario4.initialTargets[3].aisClass = 'A'
scenario4.initialTargets[4].aisClass = 'A'
scenario4.initialTargets[5].aisClass = 'A'
scenario4.initialTargets[6].aisClass = 'A'
scenario4.initialTargets[7].aisClass = 'A'
scenario4.initialTargets[8].aisClass = 'A'
scenario4.initialTargets[9].aisClass = 'A'
scenario4.initialTargets[10].aisClass = 'A'
scenario4.initialTargets[11].aisClass = 'A'
scenario4.initialTargets[12].aisClass = 'A'
scenario4.initialTargets[13].aisClass = 'A'
scenario4.initialTargets[14].aisClass = 'A'
scenario4.initialTargets[15].aisClass = 'A'

scenarioList = [scenario0, scenario1, scenario2, scenario3, scenario4]

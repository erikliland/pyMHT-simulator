import logging
import numpy as np
import time
import pymht.tracker as tomht
import pymht.utils.helpFunctions as hpf
import matplotlib.pyplot as plt
from pysimulator import simulationConfig
from pysimulator.scenarios.defaults import *
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *


def runSimulation(**kwargs):
    minToc = [float('Inf')]
    maxToc = [0]
    avgToc = []
    tic0 = time.time()

    scenario = simulationConfig.scenarioList[0]
    print(scenario.name)

    i = 4
    seed = simulationConfig.baseSeed + i - 1
    simTime = scenario.radarPeriod * 6  # sec
    lambda_phi = 2e-5
    P_d = 0.6  # Probability of detection
    N = 6  # Number of  timesteps to tail (N-scan)
    M_required = 2
    N_checks = 3
    preInitialized = kwargs.get('preInitialized', False)

    if kwargs.get('printInitialTargets', False):
        print("Initial targets:")
        print(scenario, sep='\n', end="\n\n")

    simList = scenario.getSimList(simTime=simTime)

    if kwargs.get('printSimList', False):
        print("Sim list:")
        print(*["\n".join([str(target) for target in targetList]) for targetList in simList], sep="\n\n", end="\n\n")

    scanList, aisList = scenario.getSimulatedScenario(seed, simList, lambda_phi, P_d, **kwargs)

    trackerArgs = (scenario.model,
                   scenario.radarPeriod,
                   lambda_phi,
                   scenario.lambda_nu)

    trackerKwargs = {'maxSpeedMS': maxSpeedMS,
                     'M_required': M_required,
                     'N_checks': N_checks,
                     'position': scenario.p0,
                     'radarRange': scenario.radarRange,
                     'eta2': eta2,
                     'eta2_ais': eta2_ais,
                     'N': N,
                     'P_d': P_d}

    tracker = tomht.Tracker(*trackerArgs, groundTruth=simList, **{**trackerKwargs, **kwargs})

    toc0 = time.time() - tic0
    print("Generating simulation data for {0:} targets for {1:} seconds / {2:} scans.  It took {3:.1f} ms".format(
        len(scenario), scenario.simTime, scenario.nScans, toc0 * 1000))

    if kwargs.get('printScanList', False):
        print("Scan list:")
        print(*scanList, sep="\n", end="\n\n")

    if kwargs.get('printAISList', False):
        aisList.print()

    tic1 = time.time()

    def simulate(tracker, simList, scanList, minToc, maxToc, avgToc, **kwargs):
        print("#" * 100)
        time.sleep(0.01)

        if kwargs.get('preInitialized', False):
            tracker.preInitialize(simList)

        for measurementList in scanList:
            tic = time.time()
            scanTime = measurementList.time
            aisMeasurements = aisList.getMeasurements(scanTime)

            tracker.addMeasurementList(measurementList,
                                       aisMeasurements,
                                       **kwargs)
            toc = time.time() - tic
            minToc[0] = toc if toc < minToc[0] else minToc[0]
            maxToc[0] = toc if toc > maxToc[0] else maxToc[0]
            avgToc.append(toc)
        print("#" * 100)

    if kwargs.get('profile', False):
        # try cProfile
        # try line_profiler
        # try memory_profiler
        import cProfile
        import pstats
        cProfile.runctx("simulate(tracker, simList, scanList, minToc, maxToc, avgToc, preInitialize=True)",
                        globals(), locals(), 'mainProfile.prof')
        p = pstats.Stats('mainProfile.prof')
        p.strip_dirs().sort_stats('time').print_stats(20)
        p.strip_dirs().sort_stats('cumulative').print_stats(20)
    else:
        simulate(tracker, simList, scanList, minToc, maxToc, avgToc, **kwargs)

    if kwargs.get('printTargetList', False):
        tracker.printTargetList()

    if kwargs.get('printAssociation', False):
        association = hpf.backtrackMeasurementNumbers(tracker.__trackNodes__)
        print("Association (measurement number)", *association, sep="\n")

    toc1 = time.time() - tic1

    print('Completed {0:} scans in {1:.0f} seconds. Min {2:4.1f} ms Avg {3:4.1f} ms Max {4:4.1f} ms'.format(
        scenario.nScans, toc1, minToc[0] * 1000, np.average(avgToc) * 1000, maxToc[0] * 1000))

    if 'exportPath' in kwargs:
        scenarioElement = tracker.getScenarioElement()
        scenario.storeScenarioSettings(scenarioElement)
        simList.storeGroundTruth(scenarioElement, scenario)
        variationsElement = ET.SubElement(scenarioElement, variationsTag, attrib={preinitializedTag: str(preInitialized)})
        variationDict = {pdTag: P_d,
                         lambdaphiTag: lambda_phi,
                         nTag: N,
                         mInitTag: M_required,
                         nInitTag: N_checks}

        variationElement = ET.SubElement(variationsElement,
                                         variationTag,
                                         attrib={str(k): str(v) for k, v in variationDict.items()})
        tracker._storeRun(variationElement, preInitialized, i=i, seed=seed)
        path = kwargs.get('exportPath')
        hpf.writeElementToFile(path, scenarioElement)
        print(path)
        path1 = path if preInitialized else None
        path2 = path if not preInitialized else None

        from pysimulator import scenarioAnalyzer
        scenarioAnalyzer.analyzeTrackingFile(path1)
        scenarioAnalyzer.analyzeInitFile(path2)

        from pysimulator import resultsPlotter
        resultsPlotter.plotTrackingPercentage(path1)
        resultsPlotter.plotInitializationTime(path2)
        resultsPlotter.plotRuntime(path1)

    if kwargs.get('plot'):
        import matplotlib.cm as cm
        import itertools
        colors1 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(scenario))))
        colors2 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        colors3 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        fig1 = plt.figure(num=1, figsize=(9, 9), dpi=90)
        hpf.plotRadarOutline(scenario.p0, scenario.radarRange, markCenter=False)
        # tracker.plotInitialTargets()
        # tracker.plotVelocityArrowForTrack() # TODO: Does not work
        # tracker.plotValidationRegionFromTracks() # TODO: Does not work
        desiredPlotPeriod = scenario.radarPeriod
        markEvery = max(1, int(desiredPlotPeriod / scenario.simulationTimeStep))
        hpf.plotTrueTrack(simList, colors=colors1, markevery=markEvery, label=True)
        # tracker.plotMeasurementsFromRoot(dummy=False, real = False, includeHistory=False)
        # tracker.plotStatesFromRoot(dummy=False, real=False, ais=True)
        # tracker.plotMeasurementsFromTracks(labels = False, dummy = True, real = True)
        tracker.plotLastScan()
        # tracker.plotAllScans()
        # tracker.plotLastAisUpdate()
        # tracker.plotAllAisUpdates(stepsBack=N)
        # tracker.plotHypothesesTrack(colors=colors3, markStates=True)  # CAN BE SLOW!
        tracker.plotActiveTracks(colors=colors2, markInitial=True, labelInitial=True, markRoot=False,
                                 markStates=True, real=True, dummy=True, ais=True, smooth=False)
        # tracker.plotActiveTracks(colors=colors2, markInitial=False, markRoot=False, markStates=False, real=False,
        #                          dummy=False, ais=False, smooth=True, markEnd=False)
        tracker.plotTerminatedTracks(markStates=True, real=True, dummy=True, ais=True, markInitial=True)

        plt.axis("equal")
        plt.xlim((scenario.p0[0] - scenario.radarRange * 1.05,
                  scenario.p0[0] + scenario.radarRange * 1.05))
        plt.ylim((scenario.p0[1] - scenario.radarRange * 1.05,
                  scenario.p0[1] + scenario.radarRange * 1.05))
        fig1.canvas.draw()
        plt.show()


if __name__ == '__main__':
    import argparse
    import sys
    import os

    logDir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-25s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(logDir, 'myapp.log'),
                        filemode='w')
    print(sys.version)

    parser = argparse.ArgumentParser(description="Run MHT tracker",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('-R', help="Run recursive", action='store_true')
    args = vars(parser.parse_args())
    exportPath = os.path.join(os.path.expanduser(
        '~'), 'TTK4900-Python', 'data', 'test.xml')
    print("Storing at", exportPath)
    runSimulation(plot=True,
                  profile=False,
                  printInitialTargets=True,
                  printTargetList=False,
                  printScanList=False,
                  printAssociation=False,
                  printAISList=False,
                  checkIntegrity=False,
                  exportPath=exportPath,
                  preInitialized=False,
                  localClutter=False,
                  dynamicWindow=False,
                  printSimList=False,
                  printTime=True,
                  printCluster=False,
                  pruneSimilar=False,
                  **args)
    print("-" * 100)

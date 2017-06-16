import pymht.tracker as tomht
import pymht.utils.helpFunctions as hpf
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
from pysimulator.scenarios.defaults import *
import os
import datetime


def runPreinitializedVariations(scenario, path, pdList, lambdaphiList, nList, nMonteCarlo,
                                baseSeed, **kwargs):
    changes = False
    simList = scenario.getSimList()

    try:
        scenarioElement = ET.parse(path).getroot()
    except Exception:
        scenarioElement = None

    if kwargs.get('overwrite', False) or (scenarioElement is None):
        scenarioElement = ET.Element(scenarioTag, attrib={nameTag: scenario.name})
        simList.storeGroundTruth(scenarioElement, scenario)
        scenario.storeScenarioSettings(scenarioElement)
        changes = True

    variationsElement = scenarioElement.find('.{0:}[@{1:}="True"]'.format(variationsTag, preinitializedTag))

    if variationsElement is None:
        changes = True
        variationsElement = ET.SubElement(
            scenarioElement,variationsTag,attrib={preinitializedTag: "True"})
    for P_d in pdList:
        for lambda_phi in lambdaphiList:
            for N in nList:
                variationDict = {pdTag: P_d,
                                 lambdaphiTag: lambda_phi,
                                 nTag: N,
                                 'localClutter':True,
                                 'lambda_local':scenario.lambda_local}
                variationElement = variationsElement.find(
                    '.{0:}[@{1:}="{4:}"][@{2:}="{5:}"][@{3:}="{6:}"]'.format(
                        variationTag, nTag, pdTag, lambdaphiTag, N, P_d, lambda_phi))
                if variationElement is None:
                    changes = True
                    variationElement = ET.SubElement(
                        variationsElement, variationTag,
                        attrib={str(k): str(v) for k, v in variationDict.items()})
                print(scenario.name, variationElement.attrib, nMonteCarlo, True)
                changes |= runMonteCarloSimulations(
                    variationElement, scenario, simList, nMonteCarlo, baseSeed,
                    variationDict, True, **kwargs)
    if changes:
        _renameOldFiles(path)
        hpf.writeElementToFile(path, scenarioElement)

def runInitializationVariations(scenario, path, pdList, lambdaphiList, M_N_list, nMonteCarlo,
                                baseSeed, **kwargs):
    changes = True
    simList = scenario.getSimList()

    try:
        scenarioElement = ET.parse(path).getroot()
    except Exception:
        scenarioElement = None

    if kwargs.get('overwrite', False) or (scenarioElement is None):
        changes = True
        scenarioElement = ET.Element(scenarioTag, attrib={nameTag: scenario.name})
        simList.storeGroundTruth(scenarioElement, scenario)
        scenario.storeScenarioSettings(scenarioElement)

    variationsElement = scenarioElement.find(
        '.{0:}[@{1:}="False"]'.format(variationsTag, preinitializedTag))

    if variationsElement is None:
        changes = True
        variationsElement = ET.SubElement(
            scenarioElement,variationsTag,attrib={preinitializedTag: "False"})
    for P_d in pdList:
        for lambda_phi in lambdaphiList:
            for (M, N) in M_N_list:
                variationDict = {pdTag: P_d,
                                 lambdaphiTag: lambda_phi,
                                 nTag: 6,
                                 mInitTag: M,
                                 nInitTag: N}
                variationElement = variationsElement.find(
                    '.{0:}[@{1:}="{5:}"][@{2:}="{6:}"][@{3:}="{7:}"][@{4:}="{8:}"]'.format(
                        variationTag, mInitTag, nInitTag, pdTag, lambdaphiTag, M, N, P_d, lambda_phi))
                if variationElement is None:
                    changes = True
                    variationElement = ET.SubElement(
                        variationsElement, variationTag,
                        attrib={str(k): str(v) for k, v in variationDict.items()})
                print(scenario.name, variationElement.attrib, nMonteCarlo, False)
                changes |= runMonteCarloSimulations(
                    variationElement, scenario, simList, nMonteCarlo, baseSeed,
                    variationDict, False, **kwargs)
    if changes:
        _renameOldFiles(path)
        hpf.writeElementToFile(path, scenarioElement)

def _renameOldFiles(path):
    if os.path.exists(path):
        modTime = os.path.getmtime(path)
        timeString = datetime.datetime.fromtimestamp(modTime).strftime("%d.%m.%Y %H.%M")
        head, tail = os.path.split(path)
        filename, extension = os.path.splitext(tail)
        newPath = os.path.join(head, filename + "_" + timeString + extension)
        os.rename(path, newPath)


def runMonteCarloSimulations(variationElement, scenario, simList, nSim, baseSeed,
                             variationDict, preInitialized, **kwargs):
    changes = False
    P_d = variationDict[pdTag]
    lambda_phi = variationDict[lambdaphiTag]
    N = variationDict[nTag]

    trackerArgs = (scenario.model,
                   scenario.radarPeriod,
                   lambda_phi,
                   scenario.lambda_nu)

    trackerKwargs = {'maxSpeedMS': maxSpeedMS,
                     'M_required': variationDict.get(mInitTag,M_required),
                     'N_checks': variationDict.get(nInitTag,N_checks),
                     'position': scenario.p0,
                     'radarRange': scenario.radarRange,
                     'eta2': eta2,
                     'N': N,
                     'P_d': P_d,
                     'localClutter':variationDict.get('localClutter', False),
                     'lambda_local': variationDict.get('lambda_local',2),
                     'dynamicWindow': False}
    trackersettingsElement = variationElement.find(trackerSettingsTag)
    if trackersettingsElement is None:
        changes = True
        storeTrackerData(variationElement, trackerArgs, trackerKwargs)
    if kwargs.get('printLog', True):
        print("Running scenario iteration", end="", flush=True)
    for i in range(nSim):
        runElement = variationElement.find(
        '{0:}[@{1:}="{2:}"]'.format(runTag, iterationTag,i+1))
        if runElement is not None:
            continue
        if kwargs.get('printLog', True):
            print('.', end="", flush=True)
        seed = baseSeed + i
        scanList, aisList = scenario.getSimulatedScenario(seed, simList, lambda_phi, P_d,
                                                          localClutter=False)
        changes = True
        try:
            runSimulation(variationElement, simList, scanList, aisList, trackerArgs,
                          trackerKwargs, preInitialized,seed=seed, **kwargs)
        except Exception as e:
            import traceback
            print("Scenario:", scenario.name)
            print("preInitialized", preInitialized)
            print("variationDict", variationDict)
            print("Iteration", i)
            print("Seed", seed)
            print(e)
    if kwargs.get('printLog', True):
        print()
    return changes

def runSimulation(variationElement, simList, scanList, aisList, trackerArgs,
                  trackerKwargs, preInitialized, **kwargs):

    tracker = tomht.Tracker(*trackerArgs, **{**trackerKwargs, **kwargs})

    startIndex = 1 if preInitialized else 0
    if preInitialized:
        tracker.preInitialize(simList)

    for measurementList in scanList[startIndex:]:
        scanTime = measurementList.time
        aisPredictions = aisList.getMeasurements(scanTime)
        tracker.addMeasurementList(measurementList, aisPredictions, pruneSimilar=not preInitialized)
    tracker._storeRun(variationElement, preInitialized, **kwargs)



def storeTrackerData(variationElement, trackerArgs, trackerKwargs):
    trackersettingsElement = ET.SubElement(variationElement, trackerSettingsTag)
    for k, v in trackerKwargs.items():
        ET.SubElement(trackersettingsElement, str(k)).text = str(v)
    ET.SubElement(trackersettingsElement, "lambda_phi").text = str(trackerArgs[2])
    ET.SubElement(trackersettingsElement, "lambda_nu").text = str(trackerArgs[3])

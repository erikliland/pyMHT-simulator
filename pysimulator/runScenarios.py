from pysimulator.simulationConfig import (trackingFilePathList, initFilePathList, scenarioList, pdList,
                                          lambdaphiList, nList, M_N_list, nMonteCarlo, baseSeed)
import multiprocessing as mp
from pysimulator import scenarioRunner


def trackingWorker(scenarioIndex, pdList, lambdaphiList, nList, nMonteCarlo):
    try:
        filePath = trackingFilePathList[scenarioIndex]
        print("Starting:", filePath, "Scenario index", scenarioIndex)
        scenarioRunner.runPreinitializedVariations(scenarioList[scenarioIndex],
                                                   filePath,
                                                   pdList,
                                                   lambdaphiList,
                                                   nList,
                                                   nMonteCarlo,
                                                   baseSeed,
                                                   printLog=False)
    except Exception as e:
        print(e)

def initWorker(scenarioIndex, pdList, lambdaphiList, M_N_list, nMonteCarlo):
    try:
        filePath = initFilePathList[scenarioIndex]
        print("Starting:", filePath, "Scenario index", scenarioIndex)
        scenarioRunner.runInitializationVariations(scenarioList[scenarioIndex],
                                                   filePath,
                                                   pdList,
                                                   lambdaphiList,
                                                   M_N_list,
                                                   nMonteCarlo,
                                                   baseSeed,
                                                   printLog=False)
    except Exception as e:
        print(e)


def runScenariosMultiProcessing(trackingScenarioIndices, initScenarioIndices,
                                pdList, lambdaphiList, nList, M_N_list, nMonteCarlo):
    nProcesses = 5
    pool = mp.Pool(processes=nProcesses)

    results = []
    for scenarioIndex in trackingScenarioIndices:
        results.append(pool.apply_async(trackingWorker,
                                        args=[scenarioIndex,
                                              pdList,
                                              lambdaphiList,
                                              nList,
                                              nMonteCarlo]))

    for scenarioIndex in initScenarioIndices:
        results.append(pool.apply_async(initWorker,
                                        args=[scenarioIndex,
                                              pdList,
                                              lambdaphiList,
                                              M_N_list,
                                              nMonteCarlo]))
    for r in results:
        r.get()


def runScenariosSingleProcess(trackingScenarioIndices, initScenarioIndices,
                              pdList, lambdaphiList, nList, M_N_list, nMonteCarlo):
    for scenarioIndex in trackingScenarioIndices:
        scenario = scenarioList[scenarioIndex]
        filePath = trackingFilePathList[scenarioIndex]
        print("Scenario path:", filePath)
        scenarioRunner.runPreinitializedVariations(
            scenario, filePath, pdList, lambdaphiList, nList, nMonteCarlo, baseSeed)

    for scenarioIndex in initScenarioIndices:
        scenario = scenarioList[scenarioIndex]
        filePath = initFilePathList[scenarioIndex]
        print("Scenario path:", filePath)
        scenarioRunner.runInitializationVariations(
            scenario, filePath, pdList, lambdaphiList, M_N_list, nMonteCarlo, baseSeed)


def mainSingle():
    runScenariosSingleProcess(range(len(trackingFilePathList)), range(1), pdList,
                              lambdaphiList, nList, M_N_list, nMonteCarlo)


def mainMulti():
    runScenariosMultiProcessing(range(len(trackingFilePathList)), range(1),
                                pdList, lambdaphiList, nList, M_N_list, nMonteCarlo)


if __name__ == '__main__':
    runScenariosMultiProcessing(range(len(trackingFilePathList)), range(1),
                                pdList, lambdaphiList, nList, M_N_list, nMonteCarlo)

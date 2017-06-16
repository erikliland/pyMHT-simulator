from pysimulator import scenarioAnalyzer
import multiprocessing as mp


def analyseResults(filePathList=[], initFilePathList = []):
    with mp.Pool() as pool:
        pool.map(scenarioAnalyzer.analyzeTrackingFile, filePathList)
        pool.map(scenarioAnalyzer.analyzeInitFile, initFilePathList)

if __name__ == '__main__':
    from pysimulator.simulationConfig import trackingFilePathList, initFilePathList
    analyseResults(trackingFilePathList, initFilePathList[0:1])

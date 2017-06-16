import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
from pysimulator.simulationConfig import *
from pymht.initiators.m_of_n import _solve_global_nearest_neighbour
import numpy as np

def analyzeTrackingFile(filePath):
    if filePath is None: return
    print("Starting:", filePath)
    tree = ET.ElementTree()
    try:
        tree.parse(filePath)
    except FileNotFoundError as e:
        print(e)
        return
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    # scenariosettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsList = scenarioElement.findall(variationsTag)
    # print("Scenario name:", scenariosettingsElement.find(nameTag).text)
    groundtruthList = groundtruthElement.findall(trackTag)

    for variationsElement in variationsList:
        if variationsElement.get(preinitializedTag) != "True":
            print("Deleting", variationsElement.attrib)
            scenarioElement.remove(variationsElement)
            continue
        analyzeVariationsTrackingPerformance(groundtruthList, variationsElement, acceptThreshold)
    tree.write(filePath)
    print("Done:", filePath)

def analyzeInitFile(filePath):
    if filePath is None: return
    print("Starting:", filePath)
    tree = ET.ElementTree()
    try:
        tree.parse(filePath)
    except FileNotFoundError as e:
        print(e)
        return
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    # scenariosettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsList = scenarioElement.findall(variationsTag)
    groundtruthList = groundtruthElement.findall(trackTag)

    for variationsElement in variationsList:
        if variationsElement.get(preinitializedTag) != "False":
            print("Deleting", variationsElement.attrib)
            scenarioElement.remove(variationsElement)
            continue
        analyzeVariationsInitializationPerformance(groundtruthList, variationsElement, acceptThreshold)
    tree.write(filePath)
    print("Done:", filePath)

def analyzeVariationsTrackingPerformance(groundtruthList, variationsElement, threshold):
    variationList = variationsElement.findall(variationTag)
    for variation in variationList:
        analyzeVariationTrackLossPerformance(groundtruthList, variation, threshold)

def analyzeVariationsInitializationPerformance(groundtruthList, variationsElement, threshold):
    variationList = variationsElement.findall(variationTag)
    for variation in variationList:
        analyzeVariationInitializationPerformance(groundtruthList, variation, threshold)

def analyzeVariationTrackLossPerformance(groundtruthList, variation, threshold):
    runList = variation.findall(runTag)
    for run in runList:
        estimateTrackList = run.findall(trackTag)
        matchList = _matchTrueWithEstimatedTracks(groundtruthList, estimateTrackList, threshold)
        _storeMatchList(run, matchList)

def analyzeVariationInitializationPerformance(groundtruthList, variation, threshold):
    runList = variation.findall(runTag)
    for runElement in runList:
        for initiationLogElement in runElement.findall(initializationLogTag):
            runElement.remove(initiationLogElement)
        initiationLog, falseInitiationLog = _matchAndTimeInitialTracks(groundtruthList, runElement, threshold)
        _storeInitializationLog(runElement, initiationLog, falseInitiationLog)

def _matchTrueWithEstimatedTracks(truetrackList, estimateTrackList, threshold):
    resultList = []
    for trueTrack in truetrackList:
        trueTrackStatesElement = trueTrack.find(statesTag)
        trueTrackStateList = trueTrackStatesElement.findall(stateTag)
        trueTrackID = trueTrack.get(idTag)
        trackLength = len(trueTrackStateList)
        for estimatedTrack in estimateTrackList:
            estimatedTrackID = estimatedTrack.get(idTag)
            estimatedTrackStatesElement = estimatedTrack.find(statesTag)
            smoothedEstimatedTrackStatesElement = estimatedTrack.find(smoothedstatesTag)
            estimatedStateList = estimatedTrackStatesElement.findall(stateTag)
            smoothedEstimatedStateList = smoothedEstimatedTrackStatesElement.findall(stateTag)

            timeMatch, goodTimeMatch, rmsError, lostTrack = _compareTrackList(
                trueTrackStateList,estimatedStateList,threshold)

            _, _, smoothedRmsError, _ = _compareTrackList(
                trueTrackStateList, smoothedEstimatedStateList, threshold)
            timeMatchLength = len(timeMatch)
            goodTimeMatchLength = len(goodTimeMatch)

            if goodTimeMatchLength > 0:
                trackPercent = (goodTimeMatchLength / trackLength)*100

                resultList.append((trueTrackID,
                                   estimatedTrackID,
                                   rmsError,
                                   smoothedRmsError,
                                   lostTrack,
                                   timeMatch,
                                   goodTimeMatch,
                                   timeMatchLength,
                                   goodTimeMatchLength,
                                   trackPercent))

    _multiplePossibleMatches(resultList)

    return resultList

def _matchAndTimeInitialTracks(groundtruthList, runElement, threshold):
    estimateTrackList = runElement.findall(trackTag)
    trueTrackList = []
    for trueTrack in groundtruthList:
        trueTrackStatesElement = trueTrack.find(statesTag)
        trueTrackID = trueTrack.get(idTag)
        trueTrackStateList = trueTrackStatesElement.findall(stateTag)
        for stateElement in trueTrackStateList:
            trueTrackTime = stateElement.get(timeTag)
            try:
                index = [t[0] for t in trueTrackList].index(trueTrackTime)
            except ValueError:
                trueTrackList.append((trueTrackTime, []))
                index = len(trueTrackList)-1
            trueTrackList[index][1].append((trueTrackID, stateElement))
    trueTrackList.sort(key=lambda tup: float(tup[0]))

    falseTrackIdSet = set()
    firstStateList = []
    for estimatedTrack in estimateTrackList:
        estimatedTrackID = estimatedTrack.get(idTag)
        falseTrackIdSet.add(estimatedTrackID)
        estimatedTrackStatesElement = estimatedTrack.find(statesTag)
        stateElementList = estimatedTrackStatesElement.findall(stateTag)
        if len(stateElementList) > 2:
            for s in stateElementList[1:-1]:
                estimatedTrackStatesElement.remove(s)
        firstStateElement = estimatedTrackStatesElement.find(stateTag)
        stateTime = firstStateElement.get(timeTag)
        firstStateList.append((stateTime, estimatedTrackID, firstStateElement))
    firstStateList.sort(key=lambda tup: float(tup[0]))

    initiationLog = []
    initiatedTrueTracksID = set()
    for (time, trackTupleList) in trueTrackList:
        newTracks = [s for s in firstStateList
                     if s[0] == time]
        uninitiatedTracks = [s for s in trackTupleList
                             if s[0] not in initiatedTrueTracksID]
        nNewTracks= len(newTracks)
        nUninitializedTracks = len(uninitiatedTracks)
        if nNewTracks == 0:
            continue
        deltaMatrix = np.zeros((nNewTracks, nUninitializedTracks))
        for i, initiator in enumerate(newTracks):
            for j, (id, stateElement) in enumerate(uninitiatedTracks):
                distance = np.linalg.norm(_parsePosition(initiator[2].find(positionTag)) -
                                          _parsePosition(stateElement.find(positionTag)))
                deltaMatrix[i,j] = distance
        deltaMatrix[deltaMatrix>threshold] = np.inf

        associations = _solve_global_nearest_neighbour(deltaMatrix)
        for initIndex, trackIndex in associations:
            initiatedTrueTracksID.add(uninitiatedTracks[trackIndex][0])
            falseTrackIdSet.remove(str(newTracks[initIndex][1]))
            initiationLog.append((time, uninitiatedTracks[trackIndex][0]))

        if np.any(np.all(deltaMatrix==np.inf, axis=1)): #Columns with only inf
            unresolvedNewTracks = [s for s in newTracks if s[1] in falseTrackIdSet]
            initiatedTracksList = [s for s in trackTupleList if s[0] in initiatedTrueTracksID]
            nInitiatedTracks = len(initiatedTrueTracksID)
            if nInitiatedTracks == 0:
                continue
            nUnresolvedTracks = len(unresolvedNewTracks)
            deltaMatrix2 = np.zeros((nUnresolvedTracks, nInitiatedTracks))
            for i, initiator in enumerate(unresolvedNewTracks):
                for j, (id, stateElement) in enumerate(initiatedTracksList):
                    distance = np.linalg.norm(_parsePosition(initiator[2].find(positionTag)) -
                                              _parsePosition(stateElement.find(positionTag)))
                    deltaMatrix2[i,j] = distance
            deltaMatrix2[deltaMatrix2>threshold] = np.inf
            #Crude test. Could be done as above...
            for i in range(nUnresolvedTracks):
                if np.any(deltaMatrix2[i] != np.inf):
                    falseTrackIdSet.remove(str(unresolvedNewTracks[i][1]))
            continue

    initiationLog.sort(key=lambda tup: float(tup[1]))

    falseInitiationLog = _getFalseInitiationLog(runElement, falseTrackIdSet)

    return initiationLog, falseInitiationLog

def _compareTrackList(trueTrackStateList, estimatedStateList, threshold):
    timeMatch = _timeMatch(trueTrackStateList, estimatedStateList)

    trueTrackSlice = [s
                      for s in trueTrackStateList
                      if s.get(timeTag) in timeMatch]

    estimatedTrackSlice = [s
                           for s in estimatedStateList
                           if s.get(timeTag) in timeMatch]

    assert len(timeMatch) == len(trueTrackSlice) == len(estimatedTrackSlice)

    goodTimeMatch, rmsError, lostTrack = _compareTrackSlices(
        timeMatch,trueTrackSlice,estimatedTrackSlice,threshold)

    if lostTrack is not None:
        return (timeMatch, goodTimeMatch, rmsError, lostTrack)

    return ([],[],np.nan, None)

def _compareTrackSlices(timeMatch, trueTrackSlice, estimatedTrackSlice, threshold):
    assert len(trueTrackSlice) == len(estimatedTrackSlice)
    deltaList = []
    for trueState, estimatedState in zip(trueTrackSlice, estimatedTrackSlice):
        trueTrackPosition = _parsePosition(trueState.find(positionTag))
        estimatedTrackPosition = _parsePosition(estimatedState.find(positionTag))
        delta = trueTrackPosition - estimatedTrackPosition
        deltaNorm = np.linalg.norm(delta)
        deltaList.append(deltaNorm)

    goodTimeMatch = []
    delta2List = []
    for i, (time, delta) in enumerate(zip(timeMatch, deltaList)):
        if delta > threshold:
            # print("Exceeded threshold")
            if not any([d < threshold for d in deltaList[i+1:]]):
                # print("Does not converge back")
                break
            goodIndices = [i for i, d in enumerate(deltaList[i + 1:]) if d < threshold]
            if len(goodIndices) <= 1:
                # print("Convergence is only at one time step")
                break
            elif any([d > (threshold * 2) for d in deltaList[i + 1:]]):
                # print("Future exceeds threshold*2")
                break

        goodTimeMatch.append(time)
        delta2List.append(delta**2)

    if not goodTimeMatch or not delta2List:
        return (goodTimeMatch, np.nan, None)
    meanSquaredError = np.mean(np.array(delta2List))

    if (len(goodTimeMatch) > 0) and not np.isnan(meanSquaredError):
        rmsError = np.sqrt(meanSquaredError)
        lostTrack = goodTimeMatch < timeMatch
        return (goodTimeMatch, rmsError, lostTrack)

    return ([], np.nan, None)

def _storeMatchList(run, matchList):
    for match in matchList:
        (trueTrackID,
         estimatedTrackID,
         rmsError,
         smoothedRmsError,
         lostTrack,
         timeMatch,
         goodTimeMatch,
         timeMatchLength,
         goodTimeMatchLength,
         trackPercent) = match

        trackElement = run.findall('.Track[@id="{:}"]'.format(estimatedTrackID))[0]
        statesElement = trackElement.find(statesTag)
        smoothedStatesElement = trackElement.find(smoothedstatesTag)

        trackElement.set(matchidTag, trueTrackID)
        trackElement.set(timematchTag, ", ".join(timeMatch))
        trackElement.set(goodtimematchTag, ", ".join(goodTimeMatch))
        trackElement.set(losttrackTag, str(lostTrack))
        trackElement.set(timematchlengthTag, str(timeMatchLength))
        trackElement.set(goodtimematchlengthTag, str(goodTimeMatchLength))
        trackElement.set(trackpercentTag, "{:.1f}".format(trackPercent))

        statesElement.set(rmserrorTag, "{:.4f}".format(rmsError))
        smoothedStatesElement.set(rmserrorTag, "{:.4f}".format(smoothedRmsError))

def _storeInitializationLog(runElement, initiationLog, falseInitiationLog):
    initiationLogElement = ET.SubElement(runElement, initializationLogTag)

    from collections import Counter
    count = Counter([e[0] for e in  initiationLog])
    count_list = [(i, count[i]) for i in count]
    count_list.sort(key=lambda tup: float(tup[0]))
    ET.SubElement(initiationLogElement, correctInitialTargetsTag).text = str(count_list)

    falseInitList = [(str(time),falseInitiationLog[time]) for time in sorted(falseInitiationLog)]
    ET.SubElement(initiationLogElement, falseTargetsTag).text = str(falseInitList)

def _timeMatch(trueTrackStateList, estimatedStateList):
    trueTrackTimeList = [e.get(timeTag) for e in trueTrackStateList]
    estimatedTrackTimeList = [e.get(timeTag) for e in estimatedStateList]
    commonTimes = [tT for tT in trueTrackTimeList if tT in estimatedTrackTimeList]
    return commonTimes

def _getFalseInitiationLog(runElement, falseTrackIdSet):
    falseInitiationLog = {}
    for falseTrackId in falseTrackIdSet:
        trackElement = runElement.find('Track[@id="{:}"]'.format(falseTrackId))
        statesElement = trackElement.find(statesTag)
        stateList = statesElement.findall(stateTag)
        startTime = stateList[0].get(timeTag)
        endTime = stateList[-1].get(timeTag) if trackElement.get(terminatedTag, "False") == "True" else None

        startTime = float(startTime)
        if startTime not in falseInitiationLog:
            falseInitiationLog[startTime] = [0, 0]
        falseInitiationLog[startTime][0] += 1

        if endTime is not None:
            endTime = float(endTime)
            if endTime not in falseInitiationLog:
                falseInitiationLog[endTime] = [0, 0]
            falseInitiationLog[endTime][1] -= 1
    return falseInitiationLog

def _parsePosition(positionElement):
    north = positionElement.find(northTag).text
    east = positionElement.find(eastTag).text
    position = np.array([east, north], dtype=np.double)
    return position

def _multiplePossibleMatches(resultList):
    import collections
    estimateIdList = [m[1] for m in resultList]
    duplicates = [item
                  for item, count in collections.Counter(estimateIdList).items()
                  if count > 1]
    for duplicate in duplicates:
        duplicateTuples = [(e, e[2]) for e in resultList if e[1] == duplicate]
        sortedDuplicates = sorted(duplicateTuples, key=lambda tup: tup[1])
        for e in sortedDuplicates[1:]:
            resultList.remove(e[0])

import os
import gzip
import numpy as np
from constants import constants
class recommendation:
    def __init__(self,maxRows,cosSimThreshold,minRating,recUserID):
        """
        This is the constructor function for recommendation system 
        Args:
            maxRows(int): maximum rows to be considered form the file
            cosSimThreshold(float):cosine similarity threshold, lies between -1 and 1
            minRating(float):minimum average rating to be considered computed from the ratings by similar users
            recUserID: User ID for which recommendations need to be computed
        Returns: nothing
        """
        self.utilityMatrix = dict()
        self.ratings = []
        self.reviewerID = []
        self.asin = []
        self.reviewerName = []
        self.helpful = []
        self.reviewText = []
        self.summary = []
        self.curCluster = []
        self.recUserID = recUserID # Reviewer ID of the reviewer for whom recommendation is being computed
        self.recProductID = []
        self.maxRows = maxRows
        self.cosSimThreshold = cosSimThreshold
        self.minRating = minRating
        self.recommend()
        #self.getRecError()

    def parseFile(self,filePath):
        g=gzip.open(filePath,'rb')
        for l in g:
            yield eval(l)

    def getRating(self):
        count = 0;
        for review in self.parseFile(os.path.join(os.getcwd(),constants.dataFolder,constants.electronicsPath)):
            self.ratings.append(review['overall'])
            self.reviewerID.append(review['reviewerID'])
            self.asin.append(review['asin'])
            #reviewerName.append(review['reviewerName'])
            self.helpful.append(review['helpful'])
            self.reviewText.append(review['reviewText'])
            count = count+1
            if(count == self.maxRows):
                count=count-1
                break
        self.utilityMatrix = dict.fromkeys(self.reviewerID)
        for index in range(0,count):
            if self.utilityMatrix[self.reviewerID[index]] is None:
                self.utilityMatrix[self.reviewerID[index]] = {self.asin[index]:self.ratings[index]}
            else:
                self.utilityMatrix[self.reviewerID[index]].update({self.asin[index]:self.ratings[index]})
        
    def dotProduct(self,v1,v2):
        sum = 0
        if v1 is None:
            return 0
        else:
            for key,value in v1.items():
                if  v2 is not None and v2.has_key(key) and v1[key] is not None and v2[key] is not None:
                    sum = sum + v1[key] * v2[key]
        return sum

    def isCosineSimilar(self,curUserID,randomUserID):
        curUserVector = self.utilityMatrix[curUserID]
        randomUserVector = self.utilityMatrix[randomUserID]
        cosSimNum = float(self.dotProduct(curUserVector,randomUserVector))
        cosSimDen1 = self.dotProduct(curUserVector,curUserVector)
        cosSimDen2 = self.dotProduct(randomUserVector,randomUserVector)
        if (cosSimDen1>0 and cosSimDen2>0):
            cosSim = cosSimNum/float(np.sqrt(cosSimDen1 * cosSimDen2))
        else:
            cosSim = 0
        return (cosSim  > self.cosSimThreshold)

    def getCurrentCluster(self):
        for rID in self.reviewerID:
            if(self.utilityMatrix.has_key(rID) and self.utilityMatrix.has_key(self.recUserID) and self.utilityMatrix[rID]):
                if (self.isCosineSimilar(self.recUserID,rID)):
                    self.curCluster.append(rID)
        self.curCluster = list(set(self.curCluster))
            
    def getRecommendation(self):
        allProductIDs = list(set(self.asin))
        for curProductID in allProductIDs:
            if not (self.utilityMatrix.has_key(self.recUserID) and self.utilityMatrix[self.recUserID].has_key(curProductID)):
                if(self.predictRating(curProductID) > self.minRating):
                    self.recProductID.append(curProductID)
        print "Recommendations for reviewer ID " + self.recUserID + ":"
        if len(self.recProductID) == 0:
            print "Recommendations not available."
        else:
            print self.recProductID

    def predictRating(self,recProductID):
            collectedRatings = []
            for reviewer in self.curCluster:
                    if self.utilityMatrix[reviewer].has_key(recProductID) and self.utilityMatrix[reviewer][recProductID] is not None:
                        collectedRatings.append(self.utilityMatrix[reviewer][recProductID])
            if (len(collectedRatings) > 0):
                return np.mean(collectedRatings,axis=0)
            else:
                return 0
        
    def recommend(self):
        self.getRating()
        self.getCurrentCluster()
        self.getRecommendation()

    def getRecError(self):
        self.getRating()#to get utilityMatrix from the data
        trainingSetSize = 100
        originalRatings = []
        computedRatings = []
        for trainingIndex in range(0,trainingSetSize):
            tcurRID = np.random.choice(self.utilityMatrix.keys())
            if self.utilityMatrix[tcurRID] is not None:
                tcurPID = np.random.choice(self.utilityMatrix[tcurRID].keys())
            else:
                continue
            originalRatings.append(self.utilityMatrix[tcurRID][tcurPID])
            self.utilityMatrix[tcurRID][tcurPID] = None
            self.recUserID = tcurRID
            self.getCurrentCluster()
            computedRatings.append(self.predictRating(tcurPID))
        rmseSum = 0
        for tIndex in range(0,trainingSetSize):
            if originalRatings[tIndex] is None:
                originalRatings[tIndex] = 0
            if computedRatings[tIndex] is None:
                computedRatings[tIndex]= 0
            rmseSum = rmseSum + (originalRatings[tIndex] - computedRatings[tIndex])*(originalRatings[tIndex] - computedRatings[tIndex])
        rmse = np.sqrt(float(rmseSum / trainingSetSize))
        print rmse
            
            

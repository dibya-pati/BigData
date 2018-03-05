
# coding: utf-8

# # Assignment 2 #



# import findspark
# findspark.init()
import pyspark
import random
from tifffile import TiffFile
import io
import zipfile
import numpy as np
from operator import add
from pyspark.sql import SparkSession
import operator

spark_session = SparkSession.builder.appName("Assignment2").getOrCreate()
spark_session.sparkContext.setLogLevel("WARN")

# spark_session.conf.set("spark.executor.memory", '8g')
# spark_session.conf.set('spark.executor.cores', '8')
# spark_session.conf.set('spark.cores.max', '16')
# spark_session.conf.set("spark.driver.memory",'8g')
# spark_session.conf.set("spark.driver.maxResultSize",'20g')
# spark_session.conf.set("spark.yarn.executor.memoryOverhead",'20g')
# SparkContext.setSystemProperty('spark.executor.memory', '4g')
sc=spark_session.sparkContext


# ## Step 1. Read image files into an RDD and divide into 500x500 images (25 points) ##

rdd = sc.binaryFiles('hdfs:/data/large_sample')
# rdd = sc.binaryFiles(r'/mnt/c/Users/Dibya/Downloads/BigData-Working/assignment2/a2_small_sample')

fullFileBytesRDD = rdd.map(lambda x: (x[0].split('/')[5],x[1]))
filenames = rdd.map(lambda x: x[0]).map(lambda path:path.split('/')[5]).collect()

#Q-1a
print('***************Q1****printing the filenames for Sanity check**********')
print(filenames,'\n')

def getOrthoTif(zfName, zfBytes):
 #given a zipfile as bytes (i.e. from reading from a binary file),
 # return a np array of rgbx values for each pixel
    bytesio = io.BytesIO(zfBytes)
    zfiles = zipfile.ZipFile(bytesio, "r")
    flag=0
    #find tif:
    for fn in zfiles.namelist():
        if fn[-4:] == '.tif':#found it, turn into array:
            flag=1
            tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
    return  (zfName,tif.asarray()) if flag else (None,None)

#Q-1b
imgArrayRDD=fullFileBytesRDD.map(lambda fileBytes: getOrthoTif(fileBytes[0],fileBytes[1]))


#this functions is used in naming the files with grid indexes
def createSubImg(record, factor):
    baseindex = ''
    recordnames = record[0].split('-')
    basename = recordnames[0]
    if recordnames.__len__() == 2: baseindex = int(recordnames[1])
    SubImg = []
    for num,arr in enumerate(record[1]):
        if recordnames.__len__() == 2:
            SubImg.append([basename+'-'+str(baseindex*factor+num),arr])
        else:
            SubImg.append([basename+'-'+str(num), arr])
    return SubImg


#Q1-c
#factor for creating image grids,i.e. 5 indicates dividing the grid by 5 in r & c
factor = 5
#image is the original image,factor is the factor to split,hvsplit to decide hwo to split
#hvsplit =1 on axis 0 and hvsplit=2 on axis =1
def splitImage(Image,factor,hvSplit):

    imgshape = Image.shape
    if hvSplit == 1:
        if imgshape[0] == 5000:
            return np.split(Image[:, :, :], factor * 2, axis=0)
        else:
            return np.split(Image[:, :, :], factor, axis=0)
    if hvSplit == 2:
        if imgshape[1] == 5000:
            return np.split(Image[:, :, :], factor * 2, axis=1)
        else:
            return np.split(Image[:, :, :], factor, axis=1)


imgArrayGridRDD = imgArrayRDD.filter(lambda record: record[0] is not None).\
    map(lambda record: [record[0], splitImage(record[1], factor, 1)]).\
    flatMap(lambda record: createSubImg(record, factor)).\
    map(lambda record: [record[0], splitImage(record[1], factor, 2)])

#Q1-d
imgArrayGridNamedRDD = imgArrayGridRDD.flatMap(lambda record: createSubImg(record, factor))

# sample = imgArrayGridNamedRDD.take(10)

#total image count after subimage formation
#shape
#name of 9th
# print(imgArrayGridNamedRDD.count())
# print(np.asarray(sample[0][1]).shape)
# print(sample[9][0])

#Q1-e
print('***************Q1-e**************')
filenamesToPrint = ['3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19']
print(imgArrayGridNamedRDD.filter(lambda record: record[0] in filenamesToPrint).\
      map(lambda record: record[1][0][0]).collect())


# #Q1-e Verification
# print('********************** verification from actual image coordinates *********************')
# basefilenames = [filename.split('-')[0] for filename in filenamesToPrint]
# orgFiles = imgArrayRDD.filter(lambda record: record[0] in basefilenames).collect()
# print(list(filter(lambda record: record[0].find('3677454_2025195') != -1, orgFiles))[0][1][0][0])
# print(list(filter(lambda record: record[0].find('3677454_2025195') != -1, orgFiles))[0][1][0][500])#18 if Y is axis =0
# print(list(filter(lambda record: record[0].find('3677454_2025195') != -1, orgFiles))[0][1][1500][1500])
# print(list(filter(lambda record: record[0].find('3677454_2025195') != -1, orgFiles))[0][1][1500][2000])#19 if Y is axis =0


# ## Step 2. Turn each image into a feature vector (25 points) ##
#2a
imageIntensityRDD = imgArrayGridNamedRDD.\
    map(lambda record: [record[0], (np.mean(record[1][:, :, :3], axis=2)*record[1][:, :, 3]/100).astype(int)])

# sampleimageIntensityRDD = imageIntensityRDD.take(5)
# sampleimgArrayGridNamedRDD = imgArrayGridNamedRDD.take(1)

#name of subimage
#shape of original with 4 channels
#shape of new with one mean channel
#print the mean matrix
# print(sampleimageIntensityRDD[0][0])
# print(np.asarray(sampleimgArrayGridNamedRDD[0][1]).shape)
# print(np.asarray(sampleimageIntensityRDD[3][1]).shape)
# print(sampleimageIntensityRDD[0][1])

#fit in memory if spils fit the spillover in disk
#_2 replicate on two clusters
#https://stackoverflow.com/questions/30520428/what-is-the-difference-between-memory-only-and-memory-and-disk-caching-level-in
imageIntensityRDD.persist(pyspark.StorageLevel.MEMORY_AND_DISK_2)

#2b
#this functions is used in naming the files with grid indexes
def createMeanImg(record, splitSize):

    orgShape = record[1].shape
    imgHSplit = np.asarray(np.split(record[1], orgShape[1]//splitSize, axis=1))
    imgVSplit = np.asarray(np.split(imgHSplit, orgShape[0]//splitSize, axis=1))
    meanImage = np.mean(imgVSplit.reshape(imgVSplit.shape[0], imgVSplit.shape[1], -1), axis=2)
    return [record[0], meanImage]


#splitSize is the downsampling factor, here 10 means, the new image would be 50*50
splitSize = 10
subImageIntensityRDD = imageIntensityRDD.\
    map(lambda record: createMeanImg(record, splitSize))

# print(subImageIntensityRDD.count())
# subImageIntensityRDDsample=subImageIntensityRDD.take(3)

#shape of downsampled image
#subimage name
#mean intensity vector
# print(np.asarray(subImageIntensityRDDsample[0][1]).shape)
# print(subImageIntensityRDDsample[0][0])
# print(subImageIntensityRDDsample[0][1])


#create a function and vectorize to take advantage of numpy vectorization
def discretize(element):
    if element > 1: element = 1
    elif element < -1: element = -1
    else: element = 0
    return element


vdiscretize = np.vectorize(discretize)

#Row Diff and Discretize
#2c
subImageIntensityRowDiffRDD = subImageIntensityRDD.\
    map(lambda record: [record[0], np.diff(record[1], axis=1), record[1]]).\
    map(lambda record: [record[0], vdiscretize(record[1]), record[2]])

#2d
#Column Diff and Discretize,Now it has discretized row and column matrices
subImageIntensityBothDiffRDD = subImageIntensityRowDiffRDD.\
    map(lambda record: [record[0], record[1], np.diff(record[2], axis=0)]).\
    map(lambda record: [record[0], record[1], vdiscretize(record[2])])

# subImageIntensityBothDiffRDDsample = subImageIntensityBothDiffRDD.take(1)

#print name of subimage
#print vector 1 shape
#print vector 2 shape
# print(subImageIntensityBothDiffRDDsample[0][0])
# print(subImageIntensityBothDiffRDDsample[0][1].shape)
# print(subImageIntensityBothDiffRDDsample[0][2].shape)

#function to flatten
# rowShape = subImageIntensityBothDiffRDDsample[0][1].shape
# colShape = subImageIntensityBothDiffRDDsample[0][2].shape


def appendFeatures(record):
        return [record[0], np.append(record[1].flatten(), record[2].flatten())]

#2e
#flatten and collect
imgFeaturesRDD = subImageIntensityBothDiffRDD.map(lambda record: appendFeatures(record))
imgFeaturesRDD.persist(pyspark.StorageLevel.MEMORY_AND_DISK_2)

testingSample =[1, '3678520_060192.zip-4' ,'3841528_K7B7.zip-16' ,
                '3678126_027197.zip-7' ,'3841522_K7B15.zip-15' ,'3841539_K7D2.zip-8' ,
                '3678358_045190.zip-15' ,'3678520_060192.zip-13' ,'3677454_2025195.zip-1' ,
                '3678155_030192.zip-3' ,'3678226_035200.zip-11' ,'3841519_K7B11.zip-19' ,
                '3678500_057190.zip-16' ,'3677502_2035200.zip-21' ,'3677558_2050190.zip-2' ,
                '3678256_037192.zip-15' ,'3678226_035200.zip-15','3678190_032192.zip-8' ,
                '3678125_027195.zip-2' ,'3678478_055190.zip-17' ,'3678155_030192.zip-6' ,
                '3678358_045190.zip-13' ,'3678224_035195.zip-11' ,'3678126_027197.zip-5' ,
                '3678327_042195.zip-22' ,'3678390_047190.zip-5' ,'3678257_037195.zip-12' ,
                '3678256_037192.zip-4' ,'3677454_2025195.zip-18' ,'3841522_K7B15.zip-13' ,
                '3677453_2025190.zip-6' ,'3678478_055190.zip-4' ,'3677542_2045190.zip-24' ,'3678190_032192.zip-3']

print('**********checking size*********')
# CtestRDD = imgFeaturesRDD.filter(lambda rec: rec[0] in testingSample).collect()
# for el in CtestRDD:
#     print(el[0])
#     print(el[1].shape)
# print(imgArrayGridNamedRDD.filter(lambda r: r[0] == '3841528_K7B7.zip-16').collect()[0][1].shape)
print('**********End checking size*********')

#total feature vector count= total subimages
# print(imgFeaturesRDD.count())
# imgFeaturesRDDsample = imgFeaturesRDD.take(1)
#
# #shape of features
# print(imgFeaturesRDDsample[0][1].shape)
# print(type(imgFeaturesRDDsample[0][1]))
# print(imgFeaturesRDDsample[0][1])

filenamestoprint = ['3677454_2025195.zip-1', '3677454_2025195.zip-18']
printSample = imgFeaturesRDD.filter(lambda record: record[0] in filenamestoprint).collect()

# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=1000)#defualt threshold

#2f
print('***************Q2-f**************')
print(printSample)


#3a
#create same sized chunnks by forward and reverse scanning of the vector
def createChunks(record, signaturelength=128):

    from math import ceil
    import hashlib
    signature = list()
    featurevectorsize = record[1].shape[0]
    feature = record[1].tolist()
    name = record[0]
    factorflag = featurevectorsize%signaturelength
    vectorlength = featurevectorsize//signaturelength
    
    if not factorflag:
        #create 128*38 if vector is divisible by 128
        i=0
        while i < featurevectorsize:
            md5obj = hashlib.md5()
            md5obj.update(str(feature[i:i+vectorlength]))
            signature.append(int(bin(int(md5obj.hexdigest(), 16))[64]))
            i += vectorlength
            
    else:
        #Here there are 128 digest each of length 38*2
        #in first pass it creates 64 hexdigest by running in forward direction and then 64 times in the reverse order
        #By this process no element are missed out since there is an overlap of elements
        #chunk size is vectorlength * 2, here 38*2
        i = 0
        count = 0
        while (i < featurevectorsize) and count < signaturelength//2:
            md5obj = hashlib.md5()
            md5obj.update(str(feature[i:i+vectorlength*2]).encode('ascii'))
            signature.append(int(bin(int(md5obj.hexdigest(), 16))[64]))
            i += int(vectorlength*2)
            count += 1
        
        i = 0
        count = 0
        while (i < featurevectorsize) and count < signaturelength//2:
            md5obj = hashlib.md5()
            md5obj.update(str(feature[-(1+i+vectorlength*2):-(i+1)]).encode('ascii'))
            signature.append(int(bin(int(md5obj.hexdigest(), 16))[64]))
            i += int(vectorlength*2)
            count += 1

        return [name, signature]


signaturelength = 128
imgSignatureRDD = imgFeaturesRDD.map(lambda record:createChunks(record, signaturelength))
sample = imgSignatureRDD.take(2)

#print length of signature and signature
# print(sample[0][1].__len__())
# print(sample[0][1])

#3b
#bandsize of 9,10 creates >30 similar images , so setting to 11
bandsize = 11
#bucket size should be greater than int equivalent of the bin array to ensure no clashing
#technically its same as returning the computed int for the binary array
numBuckets = 2**bandsize + 1


def computeBucket(record):
    global numBuckets
    #retun the int value of the binary array
    return [record[0], int(''.join(str(record[1]))[1:-1].replace(' ', ''), 2) % numBuckets]


#do this for each candidate in distributed fashion
def findSimilarityCount(candidate, record):

    #this matches the (minisignature,bucket) tuples of candidate with any all of such tuples in record
    #for atleast a single match it returns 1, it has matched in one of the buckets,else return 0
    for bandvalue in candidate[1]:
        if bandvalue in record[1]: return [1, record[0]]

    #if fails set the value to 0 and return
    return [0, record[0]]


#numelements is number of elements in each band(equal to bandsize when divisible)
#this is to ensure that we form uniform bands when band size doesnt divide signaturelength uniformly
#the reminder elements are ignored

numelements = (signaturelength//bandsize)*bandsize

#step1, resize the long vector to numelements * (initial_size/numelements)
#step 2,split the square matrix to band and signature
#step 3,flatten this structure,now [subimage,[band_number,mini_signature]]
#step 4,compute bucket value for each sign in each band in each subimage
#step 5,group it back with subimage name, mapValues(list) is required in python3 to expand the inner list

candidateNames = ['3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19']

imgMiniSignBucketRDD = imgSignatureRDD.\
                map(lambda record: [record[0], np.reshape(record[1][:numelements], (-1, bandsize))]).\
                map(lambda record: [record[0], [[band, sign] for band, sign in enumerate(record[1])]]). \
                flatMapValues(lambda val: val).\
                mapValues(lambda value: computeBucket(value)). \
                groupByKey().mapValues(list)

#3b
#collect the candidates for which the similar matches are to be found
candidateBucket = imgMiniSignBucketRDD.filter(lambda record: record[0] in candidateNames).collect()

# print(imgMiniSignBucketRDD.count())
# print(imgMiniSignBucketRDD.take(1))
# print(candidateBucketRDD[0])

#match with candidates , here match candidate with all elements in the RDD
#there should be at least 1 match for any band size, which is the match with self
#And there may be max all matches when the band size is 1,
#probability of atleast one match P=1-(1/2)^signaturelength, when bandsize=1
#filter matches where there is at least one match
#sum the matches for each candidate, check if its above the threshold required i.e. [10,20] matches
printcandidates = ['3677454_2025195.zip-1', '3677454_2025195.zip-18']
#this will store a dictionary of candidates and values
mtchCand = []
matchPairs = [[]]

print('***************Q3-b**************')
for candidate in candidateBucket:
    match = imgMiniSignBucketRDD.\
                    map(lambda record: findSimilarityCount(candidate, record)).\
                    filter(lambda record: record[0] == 1).\
                    map(lambda rec: rec[1]).\
                    collect()
                    # map(lambda record: record[0]).\
                    # reduce(add)
                    #.collect()

    mtchCand.extend(match)
    [matchPairs.append([candidate[0], matched]) for matched in match]
    # print(match)
    if candidate[0] in printcandidates:
    #     # print(candidate[0],printcandidates)
        print('***********'+candidate[0]+'***********')
        print(match)

# mtchCand = list(map(lambda x: x[1], mtchCand))
# print('********************** mtchCand')
# print(mtchCand)
# print('********************** matchPairs')
# print(matchPairs)

#band=10, returns one with 10 matches, (11 16 10 17 ) including one matched to self
#band=9 return 2 candidates with >30 matches (33 36 27 30)
#considering band = 9

#3c
#Compute using a sample of 10 images of the centre

#collect 10 random samples for SVD
sampleimgFeaturesMatX = imgFeaturesRDD.\
                       filter(lambda record: record[0] in mtchCand).\
                       takeSample(False, 10)

sampleimgFeaturesMat = list(map(lambda x: x[1], sampleimgFeaturesMatX))
                        # filter(lambda record: record[0] in mtchCand).\

# print('**************shapes of sample img matrix**********')
# print(len(sampleimgFeaturesMat[0]))
# print(type(sampleimgFeaturesMat[0]))
# print(np.asarray(sampleimgFeaturesMat).shape)
# print(type(sampleimgFeaturesMat))
# print(sampleimgFeaturesMat[0])

#centring and z score conversion
mu = np.mean(np.asarray(sampleimgFeaturesMat), axis=0)
std = np.std(np.asarray(sampleimgFeaturesMat), axis=0)
#set the standard deviation to 1, if its 0 to prevent math error while performing division and makes sense as well
# print(type(std))
# print(std)
std[np.where(std == 0)] = 1

sampleimgFeaturesMat_zs = (sampleimgFeaturesMat - mu) / std

U, s, Vh = np.linalg.svd(sampleimgFeaturesMat_zs, full_matrices=0, compute_uv=1)

#broadcast the Vh matrix so everyone has access to it
VhBroadcastVar = sc.broadcast(Vh.T)
# print('shape of V*******************')
# print(Vh.shape)

#compute PCA with the given Vh by multiplying each feature vector with Vh
#Vh is

def computeDist(reducedFeat, candFeatures):

    distWAllcand = [[candFeature[0], np.linalg.norm(reducedFeat-candFeature[1])] for candFeature in candFeatures]
    return distWAllcand


reducedFeatCan = imgFeaturesRDD. \
                    filter(lambda record: record[0] in candidateNames). \
                    map(lambda record: [record[0], np.dot(record[1], VhBroadcastVar.value)]).\
                    collect()

# print('shape of features of candidates********')
# print(np.asarray(reducedFeatCan[0][1]).shape)
# print('**********checking size at end*********')
# CtestRDD = imgFeaturesRDD.filter(lambda rec: rec[0] in testingSample).collect()
# for el in CtestRDD:
#     print(el[0])
#     print(el[1].shape)

imgFeaturesPCARDD = imgFeaturesRDD. \
                    filter(lambda record: record[0] in mtchCand). \
                    map(lambda record: [record[0], np.dot(record[1], VhBroadcastVar.value)]).\
                    map(lambda record: [record[0], computeDist(record[1], reducedFeatCan)]).\
                    flatMapValues(lambda val: val).\
                    map(lambda record: [[record[1][0], record[0]], record[1][1]]).\
                    filter(lambda record: record[0] in matchPairs).\
                    collect()

# print(imgFeaturesPCARDD)
print('***************Q3-c**************')
print('**********Print Eucledian Distance of similar images for 2 given images*********')

for cnd in printcandidates:
    distDict = {}
    for elem in imgFeaturesPCARDD:
        if elem[0][0] == cnd:
            distDict[elem[0][1]] = elem[1]

    sorted_distDict = sorted(distDict.items(), key=operator.itemgetter(1))
    print('***********Eucledian for '+cnd+'***********')
    print(sorted_distDict)

########################Bonus 3d###############################################################################





print('\n')
print('########################Bonus 3d#########################################################################')


# splitSize is the downsampling factor, here 10 means, the new image would be 50*50,5 100*100
splitSize = 5
subImageIntensityRDDB = imageIntensityRDD. \
    map(lambda record: createMeanImg(record, splitSize))

# Row Diff and Discretize
# 2c
subImageIntensityRowDiffRDDB = subImageIntensityRDDB. \
    map(lambda record: [record[0], np.diff(record[1], axis=1), record[1]]). \
    map(lambda record: [record[0], vdiscretize(record[1]), record[2]])

# 2d
# Column Diff and Discretize,Now it has discretized row and column matrices
subImageIntensityBothDiffRDDB = subImageIntensityRowDiffRDDB. \
    map(lambda record: [record[0], record[1], np.diff(record[2], axis=0)]). \
    map(lambda record: [record[0], record[1], vdiscretize(record[2])])


# 2e
# flatten and collect
imgFeaturesRDD = subImageIntensityBothDiffRDDB.map(lambda record: appendFeatures(record))

printSampleB = imgFeaturesRDD.filter(lambda record: record[0] in filenamestoprint).collect()

# 2f
print('\n')
# print('***************Q2-f*****Bonus**************')
# print(printSampleB)

imgSignatureRDD = imgFeaturesRDD.map(lambda record: createChunks(record, signaturelength))
sample = imgSignatureRDD.take(2)

bandsize = 11
# bucket size should be greater than int equivalent of the bin array to ensure no clashing
# technically its same as returning the computed int for the binary array
numBuckets = 2 ** bandsize + 1


# numelements is number of elements in each band(equal to bandsize when divisible)
# this is to ensure that we form uniform bands when band size doesnt divide signaturelength uniformly
# the reminder elements are ignored

numelements = (signaturelength // bandsize) * bandsize

# step1, resize the long vector to numelements * (initial_size/numelements)
# step 2,split the square matrix to band and signature
# step 3,flatten this structure,now [subimage,[band_number,mini_signature]]
# step 4,compute bucket value for each sign in each band in each subimage
# step 5,group it back with subimage name, mapValues(list) is required in python3 to expand the inner list

imgMiniSignBucketRDD = imgSignatureRDD. \
    map(lambda record: [record[0], np.reshape(record[1][:numelements], (-1, bandsize))]). \
    map(lambda record: [record[0], [[band, sign] for band, sign in enumerate(record[1])]]). \
    flatMapValues(lambda val: val). \
    mapValues(lambda value: computeBucket(value)). \
    groupByKey().mapValues(list)

# 3b
# collect the candidates for which the similar matches are to be found
candidateBucket = imgMiniSignBucketRDD.filter(lambda record: record[0] in candidateNames).collect()
# match with candidates , here match candidate with all elements in the RDD
# there should be at least 1 match for any band size, which is the match with self
# And there may be max all matches when the band size is 1,
# probability of atleast one match P=1-(1/2)^signaturelength, when bandsize=1
# filter matches where there is at least one match
# sum the matches for each candidate, check if its above the threshold required i.e. [10,20] matches
# this will store a dictionary of candidates and values

mtchCand = []
matchPairs = [[]]

# print('***************Q3-b**************')
for candidate in candidateBucket:
    match = imgMiniSignBucketRDD. \
        map(lambda record: findSimilarityCount(candidate, record)). \
        filter(lambda record: record[0] == 1). \
        map(lambda rec: rec[1]). \
        collect()
    # map(lambda record: record[0]).\
    # reduce(add)
    # .collect()

    mtchCand.extend(match)
    [matchPairs.append([candidate[0], matched]) for matched in match]
    # print(match)
    # if candidate[0] in printcandidates:
    #     #     # print(candidate[0],printcandidates)
    #     print('***********' + candidate[0] + '***********')
    #     print(match)


# collect 10 random samples for SVD
sampleimgFeaturesMatX = imgFeaturesRDD. \
    filter(lambda record: record[0] in mtchCand). \
    takeSample(False, 10)

sampleimgFeaturesMat = list(map(lambda x: x[1], sampleimgFeaturesMatX))

# centring and z score conversion
mu = np.mean(np.asarray(sampleimgFeaturesMat), axis=0)
std = np.std(np.asarray(sampleimgFeaturesMat), axis=0)

std[np.where(std == 0)] = 1

sampleimgFeaturesMat_zs = (sampleimgFeaturesMat - mu) / std

U, s, Vh = np.linalg.svd(sampleimgFeaturesMat_zs, full_matrices=0, compute_uv=1)

# broadcast the Vh matrix so everyone has access to it
VhBroadcastVar = sc.broadcast(Vh.T)


# print('shape of V*******************')
# print(Vh.shape)

# compute PCA with the given Vh by multiplying each feature vector with Vh
# Vh is



reducedFeatCan = imgFeaturesRDD. \
    filter(lambda record: record[0] in candidateNames). \
    map(lambda record: [record[0], np.dot(record[1], VhBroadcastVar.value)]). \
    collect()

imgFeaturesPCARDD = imgFeaturesRDD. \
    filter(lambda record: record[0] in mtchCand). \
    map(lambda record: [record[0], np.dot(record[1], VhBroadcastVar.value)]). \
    map(lambda record: [record[0], computeDist(record[1], reducedFeatCan)]). \
    flatMapValues(lambda val: val). \
    map(lambda record: [[record[1][0], record[0]], record[1][1]]). \
    filter(lambda record: record[0] in matchPairs). \
    collect()

# print(imgFeaturesPCARDD)
print('***************Q3-c Bonus****************')
print('**********Print Eucledian Distance of similar images for 2 given images*********')

for cnd in printcandidates:
    distDict = {}
    for elem in imgFeaturesPCARDD:
        if elem[0][0] == cnd:
            distDict[elem[0][1]] = elem[1]

    sorted_distDict = sorted(distDict.items(), key=operator.itemgetter(1))
    print('***********Eucledian for ' + cnd + '***********')
    print(sorted_distDict)








import os
from os.path import isfile, join
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import re
from operator import add
from lxml import etree
import codecs

APP_NAME = "My Spark Application"

# def create_spark_session(app_name="SparkApplication"):
#     spark_session = SparkSession.builder \
#      .appName(app_name) \
#      .master("local[*]") \
#      .getOrCreate()
#
#     spark_session.sparkContext.setLogLevel("WARN")
#
#     return spark_session

#pair in group of two elements
# def chunkOfTwo(elmnIter, n=2):
#     """Yield successive n-sized chunks from l."""
#     items=list(elmnIter)
#     for i in range(0, len(items), n):
#         yield items[i:i + n]

def IndustryCount(content,Industries):
	IndCount=list()
	# print(Industries)
	for industry in Industries:
		regString = r'(\i?)\b'+industry+r'\b'
		IndCount.append((industry,len(re.findall(regString,content,re.IGNORECASE))))
	return IndCount



def main():
	#set the Directory path Here
	dirPath= r'C:\Users\Dibya\Downloads\BigData\blogs'
	fileNameList = [filename for filename in os.listdir(dirPath) if isfile(join(dirPath,filename))]
	fileNameRDD = sc.parallelize(fileNameList)

	industryBroadcastVar = sc.broadcast(fileNameRDD.map(lambda fname: fname.split('.'))
		.map(lambda fields:(fields[3],1)).groupByKey().map(lambda item:item[0])
	     .collect())

	print(industryBroadcastVar.value)

	# dirPath=r'C:\Users\Dibya\Downloads\BigData\t1'
	fileNameList = [filename for filename in os.listdir(dirPath) if isfile(join(dirPath,filename))]
	allFilesRDD=sc.wholeTextFiles(','.join(map(lambda file:join(dirPath,file),fileNameList)))
	# print(allFilesRDD.count())


	allblogRDD = allFilesRDD.map(lambda item: item[1])
	blogSplitRDD = allblogRDD.flatMap(lambda content: content.split(r'</Blog>'))\
		.map(lambda content:re.sub(r'(?is)<Blog>\s*(\.*)', r'\1',content))\
		.map(lambda content:re.sub(r'\r\n', '',content))\
		.map(lambda content:re.sub(r'(?is)<post>(\.*)', r'\1',content))\
		.filter(lambda content: content is not '')

	postSplitRDD = blogSplitRDD.flatMap(lambda content: content.split(r'</post>'))\
		.filter(lambda content: content is not '')\
		.map(lambda content: content.split(r'</date>'))\
		.map(lambda content:(re.sub(r'(?is)<date>\s*(\.*)',r'\1',content[0]),content[1]))\
		.map(lambda content:(content[0].split(',')[2],content[0].split(',')[1],content[1]))


	postIndCountRDD=postSplitRDD\
		.map(lambda pst:((pst[0],pst[1]),IndustryCount(pst[2],industryBroadcastVar.value)))\
		.flatMapValues(lambda c:c) \
		.map(lambda item:(((item[0][0]+"-"+item[0][1],item[1][0]),item[1][1])))
	postAddIndCountRDD = postIndCountRDD.reduceByKey(add).filter(lambda item:item[1]>0)\
		.map(lambda item:(item[0][1],(item[0][0],item[1])))
	postAddIndCountGroupRDD = postAddIndCountRDD.groupByKey().mapValues(list)

	print(postAddIndCountGroupRDD.collect())

if __name__ == '__main__':
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc = SparkContext(conf=conf)
	main()

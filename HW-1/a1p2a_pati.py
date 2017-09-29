from random import random
import re
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import sys



# def create_spark_session(app_name="SparkApplication"):
#     spark_session = SparkSession.builder \
#      .appName(app_name) \
#      .master("local[*]") \
#      .getOrCreate()
#
#     spark_session.sparkContext.setLogLevel("WARN")
#
#     return spark_session


##??? take care of number of mappers and reducers to be passed to spark
class MyMapReduce:

    def __init__(self, data):  # [DONE]
        self.data = data  # the "file": list of all key value pairs
        # Configure Spark
        conf = SparkConf().setAppName(APP_NAME)
        conf = conf.setMaster("local[*]")
        self.sc = SparkContext(conf=conf)

#WordCount implementation below:
class WordCountMR(MyMapReduce):

    def countWords(self):
        wordCountRDD = self.sc.parallelize(self.data)
##removed word cleaning
##.map(lambda item: re.sub(r'(\w)[.:,]', r'\1', item)) \
        mapRet = wordCountRDD.map(lambda item:item[1].split())\
             .flatMap(lambda x:x) \
             .map(lambda word:word.lower()).map(lambda x:(x,1)).groupByKey()\
             .mapValues(lambda v:len(v))

        return mapRet


#Set Difference Implementation below:
class SetDifferenceMR(MyMapReduce):

    def findDiffElements(self):
        SetDifferenceRDD = self.sc.parallelize(self.data)
        mapRet = SetDifferenceRDD.flatMapValues(lambda x:x).map(lambda item:(item[1],item[0]))\
         .groupByKey().map(lambda item:(item[0],set(item[1])))\
         .filter(lambda item:list(item[1])[0]=='R' and len(item[1])==1).map(lambda item:item[0])

        return mapRet


def main():

    data = [
        (1, "The horse raced past the barn fell"),
        (2,
         "The complex houses married and single soldiers and their families"),
        (3, "There is nothing either good or bad, but thinking makes it so"),
        (4, "I burn, I pine, I perish"),
        (5,
         "Come what come may, time and the hour runs through the roughest day"),
        (6, "Be a yardstick of quality."),
        (7,
         "A horse is the projection of peoples' dreams about themselves - strong, powerful, beautiful"
        ),
        (8,
         "I believe that at the end of the century the use of words and general educated opinion will have altered so much that one will be able to speak of machines thinking without expecting to be contradicted."
        ), (9, "The car raced past the finish line just in time."),
        (10, "Car engines purred and the tires burned.")
    ]
    # wordCOuntRDD=sc.parallelize(data)
    WordCountMRObject = WordCountMR(data)
    print((WordCountMRObject.countWords().collect()))
    data1 = [('R', ['apple', 'orange', 'pear', 'blueberry']),
             ('S', ['pear', 'orange', 'strawberry', 'fig', 'tangerine'])]
    data2 = [('R', [x for x in range(50) if random() > 0.5]),
             ('S', [x for x in range(50) if random() > 0.75])]
    WordCountMRObject.sc.stop()
    SetDifferenceMRObject1 = SetDifferenceMR(data1)
    SetDifferenceMRObject2 = SetDifferenceMR(data2)
    print((SetDifferenceMRObject1.findDiffElements().collect()))
    print((SetDifferenceMRObject2.findDiffElements().collect()))
    SetDifferenceMRObject1.sc.stop()
    SetDifferenceMRObject1.sc.stop()



if __name__ == '__main__':
    main()

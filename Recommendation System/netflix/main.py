# coding:utf-8
from pyspark import SparkConf
from pyspark import SparkContext
if __name__ == '__main__':
    conf = SparkConf()
    conf = conf.setAppName("wordcount").setMaster("local")
    sc = SparkContext(conf=conf)
    lines = sc.textFile("word.txt", 2)
    print("lines rdd partition length = %d" % (lines.getNumPartitions()))
    result = lines.flatMap(lambda line: line.split(" ")).map(
        lambda word: (word, 1)).reduceByKey(lambda v1, v2: v1+v2, 3)
    print("result rdd partition length = %d" % (lines.getNumPartitions()))
    result.foreach(lambda x: print(x))
    result.saveAsTextFile("result")

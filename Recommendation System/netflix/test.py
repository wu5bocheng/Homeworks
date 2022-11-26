import os
import sys

os.environ['SPARK_HOME'] = "C:\spark"
sys.path.append("/opt/apache-spark-1.6.2-bin-hadoop2.7/python")
sys.path.append(
    "/opt/apache-spark-1.6.2-bin-hadoop2.7/python/lib/py4j-0.9-src.zip")
try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print("success")

except ImportError as e:
    print("error importing spark modules", e)
    sys.exit(1)

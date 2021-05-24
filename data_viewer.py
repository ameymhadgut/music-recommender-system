#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
import argparse

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, lit

def main(spark, netID, filename = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet', num_of_rows = 10):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print(filename)
    songs = spark.read.parquet(filename)
    songs.show(num_of_rows)
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('project1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help = 'Enter file name for dataset in parquet format', default = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    parser.add_argument('--numOfRows', help = 'Enter number of rows to be printed', default = 10)
    args = parser.parse_args()

    # Call our main routine
    main(spark, netID, args.filename, int(args.numOfRows))

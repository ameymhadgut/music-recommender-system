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

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    #Load the original full parquet file
    msd_path1 = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    msd_path2 = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    msd_path3 = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'

    #local_path = 'hdfs:/user/arm994/project-data/cf_train.parquet'
    #run_path = local_path
    # Load the full parquet data into DataFrame
    songs1 = spark.read.parquet(msd_path1)
    songs2 = spark.read.parquet(msd_path2)
    songs3 = spark.read.parquet(msd_path3)
    
    print('Unique users count: ', songs1.select('user_id').distinct().count())
    print('Unique users count: ', songs2.select('user_id').distinct().count())
    print('Unique users count: ', songs3.select('user_id').distinct().count())

    '''
    print('Read the parquet')
    print('Summary of the dataset: ', run_path)
    print('Song dataframe count: ',songs.count())
    print('Unique users count: ', songs.select('user_id_num').distinct().count())
    print('Max count : ', songs.agg({"count": "max"}).collect()[0][0])
    print('Min count : ', songs.agg({"count": "min"}).collect()[0][0])
    '''
    
    '''
    Read the parquet
    Summary of the dataset: hdfs:/user/bm106/pub/MSD/cf_train.parquet
    Song dataframe count: 49824519
    Unique users count: 1129318
    Max count : 9667
    Min count : 1

    Read the parquet
    Summary of the dataset:  hdfs:/user/arm994/project-data/cf_train_RS_1.parquet
    Song dataframe count:  498425
    Unique users count:  334590
    Max count :  664
    Min count :  1


    Train - Unique users count:  1129318
    Validation - Unique users count:  10000
    Test - Unique users count:  100000

    '''

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('project1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)

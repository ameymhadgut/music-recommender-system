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

def main(spark, netID, file_name = 'cf_train', fraction = 0.01, filterCount1 = True):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    #Load the original full parquet file
    msd_path = 'hdfs:/user/bm106/pub/MSD/'
    local_path = 'hdfs:/user/arm994/project-data/'
    ext = '.parquet'

    # Load the full parquet data into DataFrame
    songs = spark.read.parquet(msd_path + file_name + ext)

    # Removing interactions with count <= 1
    if(filterCount1):
        songs = songs.filter(songs['count'] <= 1)
    
    print('Read the parquet')
    #print('Song dataframe count: ?',songs.count())

    # Getting unique user_id's to use them in fraction for sampleBy
    unique_user_ids = songs.select("user_id").distinct()
    unique_user_ids = unique_user_ids.withColumn("fraction", lit(fraction))
    print('Got unique user_id\'s')

    #print('Top 10 user_ids:')
    #unique_user_ids.show(10)

    # Creating a map {user_id: fraction} to use in fractions for sampleBy
    fractionMap = unique_user_ids.select("user_id", "fraction").rdd.collectAsMap()

    print('Got fraction map')
    #print('Top 10 from fractionMap:')
    #print(list(fractionMap.items())[:10])

    # Getting only half of the unique user_ids since full list slows down, targeting 50% of users at least
    fraction_list = list(fractionMap.items())
    fractionMap_new = dict(fraction_list[:len(fraction_list)//2])

    print('Got fractionMap_new map')

    # Sampling the dataset using stratified sampling ensuring at least 50% unique user's are included in subsample
    sampled_DF = songs.sampleBy("user_id", fractionMap_new, None)
    print('Sampled the parquet')
    
    #print(sampled_DF.count())
    #print('Top 10 user_ids:')
    #sampled_DF.show(10) #1% = 499308
    
    '''
    # Random Sample fraction of the data
    sampled_DF = songs.sample(False, fraction, None)
    '''

    # Write sampled subset to parquet
    sampled_DF.write.mode("overwrite").parquet(local_path + file_name + '_' + str(int(fraction*100)) + ext)
    print('Wrote to the output parquet')
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('project1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help = 'Enter file name for dataset in parquet format', default = 'cf_train')
    parser.add_argument('--fraction', help = 'Enter fraction of dataset to sample', default = '0.01')
    parser.add_argument('--filterCount1', help = 'Enter True/False to filter out count 1 interactions', default = True)
    args = parser.parse_args()

    # Call our main routine
    main(spark, netID, args.filename, float(args.fraction), args.filterCount1)

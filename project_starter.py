#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
# Use getpass to obtain user netID
import getpass

# Import PySpark stuff
from pyspark.sql import SparkSession
from pyspark.sql import functions as fn
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType

# Import Utilities
import itertools
import random

def readData(train_file, validation_file, test_file):
    '''
    Read parquet files and return dataframes for train, validation and test dataset
    input : train_file, validation_file, test_file => file names with path for parquet files
    output: Returns train, validation and test dataframes
    '''

    # Read parquet
    train_df = spark.read.parquet(train_file)
    validation_df = spark.read.parquet(validation_file)
    test_df = spark.read.parquet(test_file)

    # Return Dataframe
    return train_df, validation_df, test_df


def sampleByUser(train_df, validation_df, fraction_users = 0.1):
    '''
    Returns subsampled training set which consists of a subset of users [all users in validation set + random_sample(users in train set)]
    input : train_df, validation_df => train and validation dataframe
            fraction_users => fraction of delta users needed, default = 0.1
    output: Returns subsampled train dataframe
    '''

    # Users in train set
    users_in_trainset = set([row['user_id'] for row in train_df.select('user_id').distinct().collect()])
    
    # Users in validation set
    users_in_valset = set([row['user_id'] for row in validation_df.select('user_id').distinct().collect()])
    
    # Users in trainset but not in validation set
    delta_users = list(users_in_trainset - users_in_valset)

    # Get fraction of random users 
    num_of_users = int(len(delta_users) * fraction_users)
    print('You got ', num_of_users, ' users in here')

    # Getting random out of list
    user_random_df = random.sample(delta_users, num_of_users)

    total_users = list(users_in_valset)
    total_users += user_random_df
    train_df = train_df.where(train_df.user_id.isin(total_users))
    
    return train_df


def preprocess(train_df, validation_df, test_df):
    '''
    Returns dataframes with numeric column values for user_id and track_id generated using StringIndexer 
            converts user_id => user_id_num and track_id => track_id_num
    input : train_df, validation_df, test_df => train, validation and test dataframes
    output: Returns indexed dataframes
    '''

    # Using StringIndexer
    userid_indexer = StringIndexer(inputCol='user_id', outputCol='user_id_num', handleInvalid = 'skip')
    userid_indexer_model = userid_indexer.fit(train_df)

    trackid_indexer = StringIndexer(inputCol='track_id', outputCol='track_id_num', handleInvalid='skip')
    trackid_indexer_model = trackid_indexer.fit(train_df)

    train_df = userid_indexer_model.transform(train_df)
    train_df = trackid_indexer_model.transform(train_df)
    #train_df.show(10)

    validation_df = userid_indexer_model.transform(validation_df)
    validation_df = trackid_indexer_model.transform(validation_df)
    #validation_df.show(10)

    test_df = userid_indexer_model.transform(test_df)
    test_df = trackid_indexer_model.transform(test_df)
    #test_df.show(10)

    return train_df, validation_df, test_df


def tune_model(train_df, validation_df, params):
    '''
    Spark Pipeline to tune model against validation set with multiple params
    input : train_df, validation_df => train and validation dataframes
            params => dict with list of values for each hyperparameters { 'rank': list(), 'regParam' = list(), 'alpha' = list() } 
    output: None
    '''

    # Pick out users from validation set
    user_id = validation_df.select('user_id_num').distinct()
    ground_truth = validation_df.select('user_id_num', 'track_id_num').groupBy('user_id_num').agg(fn.expr('collect_list(track_id_num) as true_item'))

    for param in itertools.product(params['rank'], params['regParam'], params['alpha']):
        print('******************** START *******************************')
        print('Started training.. Parameters are {}'.format(param))

        # Config ALS
        als = ALS(userCol = 'user_id_num', itemCol = 'track_id_num', ratingCol = 'count', implicitPrefs = True, nonnegative = True, coldStartStrategy = "drop", numUserBlocks = 10, numItemBlocks = 10, checkpointInterval = 5)
        als.setParams(maxIter = 10, rank = param[0], regParam = param[1], alpha = param[2])

        # Train model
        model = als.fit(train_df)

        print('Done Training.. Parameters are {}'.format(param))

        print('Started getting predictions.. Parameters are {}'.format(param))

        # Getting 500 recommendations per user for validation set
        res = model.recommendForUserSubset(user_id, 500)
        predicted = res.select('user_id_num','recommendations.track_id_num')

        # Join ([predicted], [ground truth])
        predictionAndLabels = predicted.join(fn.broadcast(ground_truth), 'user_id_num', 'inner').rdd.map(lambda row: (row[1], row[2]))
        
        print('Done getting predictions.. Parameters are {}'.format(param))

        print('Started eval.. Parameters are {}'.format(param))

        # Ranking metrics eval
        metrics = RankingMetrics(predictionAndLabels)
        
        precAt = metrics.precisionAt(500)
        mAP = metrics.meanAveragePrecision
        
        print('Evaluation Completed.. Parameter are {} '.format(param)) 
        print('Results are:\nPrecision at k = 500 is {} \nMean Average Precision is {}'.format(precAt, mAP))
        print('******************** END *******************************')


def final_model(train_df, test_df, param):
    '''
    Spark Pipeline to test final model against test set
    input : train_df, test_df => train and test dataframes
            param => dict with values for each hyperparameters { 'rank': val, 'regParam' = val, 'alpha' = val } 
    output: Returns model
    '''

    # Pick out users from validation set
    user_id = test_df.select('user_id_num').distinct()
    ground_truth = test_df.select('user_id_num', 'track_id_num').groupBy('user_id_num').agg(fn.expr('collect_list(track_id_num) as true_item'))


    print('******************** START *******************************')
    print('Started training.. Parameters are {}'.format(param))

    # Config ALS
    als = ALS(userCol='user_id_num', itemCol='track_id_num', ratingCol='count', implicitPrefs=True, nonnegative=True, coldStartStrategy="drop", numUserBlocks = 10, numItemBlocks = 10, checkpointInterval = 5)
    als.setParams(maxIter = 10, rank = param['rank'], regParam = param['regParam'], alpha = param['alpha'])

    # Train model
    model = als.fit(train_df)

    print('Done Training.. Parameters are {}'.format(param))

    print('Started getting predictions.. Parameters are {}'.format(param))

    # Getting 500 recommendations per user for validation set
    res = model.recommendForUserSubset(user_id, 500)
    predicted = res.select('user_id_num','recommendations.track_id_num')

    # Join ([predicted], [ground truth])
    predictionAndLabels = predicted.join(fn.broadcast(ground_truth), 'user_id_num', 'inner').rdd.map(lambda row: (row[1], row[2]))
        
    print('Done getting predictions.. Parameters are {}'.format(param))

    print('Started eval on test set.. Parameters are {}'.format(param))

    # Ranking metrics eval
    metrics = RankingMetrics(predictionAndLabels)
        
    precAt = metrics.precisionAt(500)
    mAP = metrics.meanAveragePrecision
        
    print('Evaluation Completed on test set.. Parameter are {} '.format(param)) 
    print('Results are:\nPrecision at k = 500 is {} \nMean Average Precision is {}'.format(precAt, mAP))
    print('******************** END *******************************')

    return model

def saveModelFactors(model, dataframe):
    '''
    Save user vectors to csv file for using with annoy
    input : model => model with latent factors
            dataframe => dataframe with values for vector
    output: None
    '''

    print('Saving vectors...')
    filename1 = 'hdfs:/user/arm994/project-data/user_factors.csv'
    filename2 = 'hdfs:/user/arm994/project-data/values.csv'

    userfactor_df = model.userFactors
    uf_df = userfactor_df.select('features')
    uf_df = uf_df.withColumn("features", fn.col("features").cast(StringType()))
    uf_df.coalesce(1).write.mode("overwrite").csv(filename1, header=True) 

    userfactor_df = userfactor_df.withColumnRenamed("id","user_id_num")
    user_id = dataframe.select('user_id_num').distinct()
    values_df = user_id.join(userfactor_df, 'user_id_num','inner').select('features')
    values_df = values_df.withColumn("features", fn.col("features").cast(StringType()))
    values_df.coalesce(1).write.mode("overwrite").csv(filename2, header=True) 

    print('Saved vectors...')

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    
    # Reading files
    base_path = 'hdfs:/user/bm106/pub/MSD/'
    train_file = base_path + 'cf_train.parquet'
    validation_file = base_path + 'cf_validation.parquet' 
    test_file = base_path + 'cf_test.parquet'
   
    # Get dataframes
    train_df, validation_df, test_df = readData(train_file, validation_file, test_file)
    
    # Sample down data by number of users
    train_df = sampleByUser(train_df, validation_df, 0.3)
    
    # Convert IDs from Alphanumeric to Numeric
    train_df, validation_df, test_df = preprocess(train_df, validation_df, test_df)
    
    #params = { 'rank': [10, 15, 20, 30], 'regParam' : [0.1, 0.5, 1], 'alpha' : [1, 5, 10] }
    #tune_model(train_df, validation_df, params)

    # Testing best model 
    param = { 'rank': 20, 'regParam' : 0.1, 'alpha' : 10 }
    model = final_model(train_df, test_df, param)
   
    # Saving latent factors for Annoy
    saveModelFactors(model, test_df)
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('project1').getOrCreate()
    spark.conf.set("spark.blacklist.enabled", False)

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)

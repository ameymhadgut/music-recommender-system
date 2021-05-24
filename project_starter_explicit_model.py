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
from pyspark.sql import functions as fn
import pandas as pd 

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import DoubleType

def train_model(training_df, validation_df, params, evaluator):
    # Use Spark's alternating least squares (ALS) method
    # Init ALS from pyspark ML module
    als = ALS()

    # Now set the parameters for the method
    als.setMaxIter(5).setItemCol("track_id_num").setRatingCol("count").setUserCol("user_id_num") 
    
    # Set the hyper params
    als.setParams(rank = params['rank'], regParam = params['regParam'])

    # Build the model with the hyper params
    model = als.fit(training_df)

    # Run the model to create a prediction and validate
    results_df = model.transform(validation_df)

    # Filter out NaN values from predictions (due to SPARK-14489)
    prediction_df = results_df.filter(results_df.prediction != float('nan'))
    prediction_df = prediction_df.withColumn("prediction", fn.abs(fn.round(prediction_df["prediction"],0)))
    
    # Get Root Mean Square Error using RSME evaluator in RegressionEvaluator the predictions
    rmse_err = evaluator.evaluate(prediction_df)

    return prediction_df, model, rmse_err

def test_model(test_df, model, evaluator):
    # Test the model against test dataset
    test_df = test_df.withColumn("count", test_df["count"].cast(DoubleType()))
    result_df = model.transform(test_df)

    # Remove NaNs
    clean_result_df = result_df.filter(result_df.prediction != float('nan'))

    # Round floats to whole numbers
    clean_result_df = clean_result_df.withColumn("prediction", fn.abs(fn.round(clean_result_df["prediction"],0)))

    # Run the previously created RMSE evaluator, on the predicted_test_df DataFrame
    RMSE_for_test = evaluator.evaluate(clean_result_df)

    return RMSE_for_test

def train_best_model(training_df, validation_df, ranks, regParams, evaluator):
    min_rmse_error = float('inf')
    best_rank = ranks[0]
    best_reg_param = regParams[0]
    best_model = None
    final_result_df = None

    for currRegParam in regParams:
      for currRank in ranks:
        params = { 
            'rank' : currRank, 
            'regParam' : currRegParam 
        }
        prediction_df, model, rmse_err = train_model(training_df, validation_df, params, evaluator)
        if rmse_err < min_rmse_error:
            min_rmse_error = rmse_err
            best_rank = currRank
            best_reg_param = currRegParam
            best_model = model
            final_result_df = prediction_df
    
    return final_result_df, best_model, best_rank, best_reg_param, min_rmse_error


def get_suggestions(user_id, songs_df, model):
    user_tracks = songs_df.filter(songs_df.user_id_num == user_id).select('track_id', 'track_id_num')
                                              
    # generate list of listened songs
    user_playlist = []
    for song in user_tracks.collect():
      user_playlist.append(song['track_id_num'])

    print('Track history for :', user_id)
    user_tracks.show(5)

    # generate dataframe of unlistened songs
    unknown_tracks = songs_df.filter(~ fn.col('track_id_num').isin(user_playlist)).distinct()
    unknown_tracks.show(5)

    # feed unlistened songs into model
    predicted_tracks = model.transform(unknown_tracks)

    # remove NaNs
    predicted_tracks = predicted_tracks.filter(predicted_tracks['prediction'] != float('nan'))

    # print output
    predicted_tracks = predicted_tracks.select('track_id').distinct().orderBy('prediction', ascending = False)
    print('Predicted Songs:')
    predicted_tracks.show(10)
    return predicted_tracks

def preprocess_data(songs_df, print_5results = True):
    # Change user id and track id from alphanumeric to integer in a new column *_num using StringIndexer()
    # user_id (alphanumeric) => user_id_num (numeric)
    indexer = StringIndexer(inputCol="user_id", outputCol="user_id_num")
    user_id_num_df = indexer.fit(songs_df).transform(songs_df)
    if print_5results:
        user_id_num_df.show(5)

    # track_id (alphanumeric) => track_id_num (numeric)
    indexer = StringIndexer(inputCol="track_id", outputCol="track_id_num")
    track_id_num_df = indexer.fit(user_id_num_df).transform(user_id_num_df)
    if print_5results:
        track_id_num_df.show(5)
    
    return track_id_num_df

def build_initial_model(filepath):
    # Read the train set
    # Columns: user_id, count, track_id
    songs_df = spark.read.parquet(filepath)
    
    # Preprocessing data
    songs_df = preprocess_data(songs_df)
    
    # Spilt dataset - 60% training, 20% validation, 20% testing
    (training_df, validation_df, test_df) = songs_df.randomSplit([0.6, 0.2, 0.2], None)

    # Print dataset stats
    print('Training: {0}, validation: {1}, test: {2}\n'.format(training_df.count(), validation_df.count(), test_df.count()))
    training_df.show(3)
    validation_df.show(3)
    test_df.show(3)
    
    validation_df = validation_df.withColumn("count", validation_df["count"].cast(DoubleType()))

    # Training the model
    params = { 'rank' : 10, 'regParam' : 0.4 }

    # Evalutor 
    regression_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")

    prediction_df, model, rmse_err = train_model(training_df, validation_df, params, regression_eval)
    print('Model Params: Rank: {0}, Regularization parameter: {1}\n'.format(params['rank'], params['regParam']))
    print('Model Evaluation: RMSE: {0}\n'.format(rmse_err))
    prediction_df.show(10)
    
    # Test the model against test dataset, TO DO: Test using average method.
    test_RMSE = test_model(test_df, model, regression_eval)
    print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

    #predicted_tracks = get_suggestions(13, songs_df, model)
    
    return model

def build_best_model(filepath):
    # Read the train set
    # Columns: user_id, count, track_id
    songs_df = spark.read.parquet(filepath)
    
    # Preprocessing data
    songs_df = preprocess_data(songs_df)
    
    # Spilt dataset - 60% training, 20% validation, 20% testing
    (training_df, validation_df, test_df) = songs_df.randomSplit([0.6, 0.2, 0.2], None)

    # Print dataset stats
    print('Training: {0}, validation: {1}, test: {2}\n'.format(training_df.count(), validation_df.count(), test_df.count()))
    training_df.show(3)
    validation_df.show(3)
    test_df.show(3)
    
    validation_df = validation_df.withColumn("count", validation_df["count"].cast(DoubleType()))

    # Evalutor 
    regression_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")

    # Train for multiple hyper params to get best model
    ranks = [8, 10, 12, 16, 20]
    regParams = [0.20, 0.25, 0.30, 0.40]
    prediction_df, best_model, best_rank, best_reg_param, rmse_error = train_best_model(training_df, validation_df, ranks, regParams, regression_eval)
    print('Best Model Params: Rank: {0}, Regularization parameter: {1}\n'.format(best_rank, best_reg_param))
    print('Model Evaluation: RMSE: {0}\n'.format(rmse_error))
    prediction_df.show(10)
    
    
    # Test the model against test dataset, TO DO: Test using average method.
    test_RMSE = test_model(test_df, best_model, regression_eval)
    print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

    #predicted_tracks = get_suggestions(13, songs_df, best_model)

    return best_model

def main(spark, netID, filepath):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    base_path = 'hdfs:/user/bm106/pub/MSD/'
    train_filename = base_path + 'cf_train.parquet'
    validation_filename = base_path + 'cf_validation.parquet'
    test_filename = base_path + 'cf_test.parquet'

    # Read data
    train_df = spark.read.parquet(train_filename)
    validation_df = spark.read.parquet(validation_filename)
    test_df = spark.read.parquet(test_filename)

    # Preprocessing data
    train_df = preprocess_data(train_df)
    validation_df = preprocess_data(validation_df)
    test_df = preprocess_data(test_df)

    # Training the model
    params = { 'rank' : 10, 'regParam' : 0.4 }

    # Evalutor 
    regression_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")

    prediction_df, model, rmse_err = train_model(train_df, validation_df, params, regression_eval)
    print('Model Params: Rank: {0}, Regularization parameter: {1}\n'.format(params['rank'], params['regParam']))
    print('Model Evaluation: RMSE: {0}\n'.format(rmse_err))
    prediction_df.show(10)
    
    # Test the model against test dataset
    test_RMSE = test_model(test_df, model, regression_eval)
    print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

    #predicted_tracks = get_suggestions(13, songs_df, model)
    


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('project1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', help = 'Enter file path for dataset in parquet format', default = 'hdfs:///user/arm994/project-data/cf_train_1.parquet')
    args = parser.parse_args()

    # Call our main routine
    main(spark, netID, args.filepath)

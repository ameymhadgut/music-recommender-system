# music-recommender-system
Music recommender system based on collaborative filtering and search query optimization using trees

## Dataset:
On HDFS, we had the count data (implicit data) in the format (user_id, count, track_id) spilt into train, validation, and test dataset adopted from http://millionsongdataset.com/ 
No. of Unique Users per set -
Train set:	1129318
Validation set:	10000
Test set:	100000

## Part 1 - Recommendation based on collaborative filtering using Alternating Least Square (ALS) module from PySpark’s ML module
Build an implicit model, tuned it using the 3 key hyperparameters (rank, regParam, Alpha) and got the top 500 recommendations per user. Evaluated the model using Mean Average Precision to ensure the ranking is tested against the ground truth set. More details in Project Report.pdf.

![image](https://user-images.githubusercontent.com/18590547/119394316-06372a80-bca0-11eb-8b34-0b52dfdd2007.png)
![image](https://user-images.githubusercontent.com/18590547/119394326-09cab180-bca0-11eb-8483-784acbecd578.png)
![image](https://user-images.githubusercontent.com/18590547/119394336-0d5e3880-bca0-11eb-9125-94c23940c7e9.png)


## Part 2 - Fast Search using Spotify's annoy
Used Annoy to improve the query search performance and compared it against linear exhaustive search. Used dot distance to find similarity. 

![image](https://user-images.githubusercontent.com/18590547/119394079-cb34f700-bc9f-11eb-8aa4-be64fd44ebee.png)
![image](https://user-images.githubusercontent.com/18590547/119394104-cff9ab00-bc9f-11eb-92b1-bb7c12887c90.png)

Also, gauged the recall and time complexity across varying number of trees in annoy.

![image](https://user-images.githubusercontent.com/18590547/119394161-e0aa2100-bc9f-11eb-930f-bccbab357266.png)
![image](https://user-images.githubusercontent.com/18590547/119394184-e56ed500-bc9f-11eb-93c6-dac505dca75b.png)

More details in the report.

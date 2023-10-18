### CS 777 Project: Clash of Clans
# Jia Liang Ma

# Packages import
import os
import sys
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, udf
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline

import matplotlib.pyplot as plt


#df = pd.read_csv('coc_clans_dataset.csv')
#print(df.head(5))

# Helper function to compute the mode for a given column
def compute_mode(df, column):
    return df.groupBy(column).count().orderBy('count', ascending=False).collect()[0][0]

def clean_data(df):
    # Convert isFamilyFriendly to binary (1 = True and 0 = False)
    df = df.withColumn('isFamilyFriendly', (F.col('isFamilyFriendly') == 'True').cast(IntegerType()))
    
    # Convert clan_type to binary (1 = open and 0 = closed)
    df = df.withColumn('clan_type', (F.col('clan_type') == 'open').cast(IntegerType()))
    
    # Drop unuseful columns
    columns_to_drop = ['clan_tag', 'clan_name', 'clan_description', 'clan_badge_url', 'war_frequency', 'clan_war_league', 'capital_league', 'clan_location']
    df = df.drop(*columns_to_drop)
    
    # Convert columns from string to int data types
    columns_to_convert = [
        'clan_level', 'clan_points', 'clan_builder_base_points', 'clan_versus_points', 'required_trophies',
        'war_win_streak', 'war_wins', 'war_ties', 'war_losses', 'num_members', 'required_builder_base_trophies',
        'required_versus_trophies', 'required_townhall_level', 'clan_capital_hall_level', 'clan_capital_points',
        'mean_member_level', 'mean_member_trophies'
    ]

    for column in columns_to_convert:
        df = df.withColumn(column, df[column].cast(IntegerType()))

    # For numerical columns, impute with mean
    # Using this to maintain the general distribution
    for column in columns_to_convert:
        mean_value = df.agg({column: 'mean'}).collect()[0][0]
        df = df.na.fill({column: mean_value})

    # For categorical columns, impute with mode
    # Using most common category to replace the null value
    categorical_columns = ['isFamilyFriendly', 'clan_type']
    for column in categorical_columns:
        mode_value = compute_mode(df, column)
        df = df.na.fill({column: mode_value})

    return df


### Load data (increased kyro size)
spark = SparkSession.builder \
    .appName("ClashOfClans_FinalProject") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate() # We test for different value since running out of java heap space


df = spark.read.csv(sys.argv[1], header=True, inferSchema=True)
#df = spark.read.csv("coc_clans_dataset.csv", header=True, inferSchema=True)



### Exploratory data analysis (EDA)
# Data display
print(df.head())

# Overview of df
df.printSchema()



### Data cleaning
# Visualize missing data for "clan_location" column
missing_count = df.filter(F.col("clan_location").isNull()).count()
non_missing_count = df.count() - missing_count

label = ['Missing value', 'Non-missing value']
size = [missing_count, non_missing_count]
color = ['lightblue', 'lightgreen']
explode = (0.1, 0) # Explode the first slice for emphasis

plt.pie(size, explode=explode, labels=label, colors=color, autopct='%1.1f%%', startangle=120)
plt.title('Missing value in "clan_location" column')
plt.axis('equal')
#plt.show()

# Clean df
df = clean_data(df)

# Check for null values in the entire dataframe
for column in df.columns:
    null_count = df.filter(F.col(column).isNull()).count()
    if null_count > 0:
        print(f"{column} has {null_count} null values")

# Impute or remove null values
# Example: Remove rows with any null values
df = df.na.drop()

# Store the cleaned df for later use
df.cache()
#df.count()


### Data preprocessing
# Convert string cols to indexed numeric cols, excluding target cols
# Handle unseen labels using keep (debug point)
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep").fit(df) 
            for column in list(set(df.columns)-set(['mean_member_trophies', 'mean_member_level']))]

pipeline = Pipeline(stages=indexers)
df_indexed = pipeline.fit(df).transform(df)

# Transform the data using only the VectorAssembler
feature_columns = [col for col in df_indexed.columns if col not in ['clan_type', 'mean_member_trophies', 'mean_member_level']]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid='skip')
assembled_data = assembler.transform(df_indexed)

# UDF to extract the 31st element from the feature vector
@udf(IntegerType())
def extract_feature(vector):
    return int(vector[31])

# Register and use the UDF to extract the 31st element from the feature vector
assembled_data = assembled_data.withColumn("problematic_feature", extract_feature("features"))

# Show the count of unique values for the problematic feature
problematic_feature_count = assembled_data.select("problematic_feature").distinct().count()
print(f"The problematic feature has {problematic_feature_count} unique values.")



spark.stop()
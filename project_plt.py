import os
import sys
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher

import matplotlib.pyplot as plt


### Load data (increased kyro size)
spark = SparkSession.builder \
    .appName("ClashOfClans_FinalProject") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate() # We test for different value since running out of java heap space


df = spark.read.csv('coc_clans_dataset.csv', header=True, inferSchema=True)
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
plt.show()

spark.stop()
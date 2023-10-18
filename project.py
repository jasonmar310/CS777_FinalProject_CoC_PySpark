### CS 777 Project: Clash of Clans
# Jia Liang Ma

# Packages import
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


#df = pd.read_csv('coc_clans_dataset.csv')
#print(df.head(5))

# Helper function to compute the mode for a given column
def compute_mode(df, column):
    return df.groupBy(column).count().orderBy('count', ascending=False).collect()[0][0]

def clean_data(df):
    # Convert isFamilyFriendly to binary (1 = True, 0 = False)
    df = df.withColumn('isFamilyFriendly', (F.col('isFamilyFriendly') == 'True').cast(IntegerType()))
    
    # Convert clan_type to binary (1 = open, 0 = closed)
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

# Helper function to extract TN, TP, FN, FP from cm
def extract_confusion_metrics(confusion_matrix):
    tn = next((row['count'] for row in confusion_matrix if row['clan_type_index'] == 0 and row['prediction'] == 0), 0)
    fp = next((row['count'] for row in confusion_matrix if row['clan_type_index'] == 0 and row['prediction'] == 1), 0)
    fn = next((row['count'] for row in confusion_matrix if row['clan_type_index'] == 1 and row['prediction'] == 0), 0)
    tp = next((row['count'] for row in confusion_matrix if row['clan_type_index'] == 1 and row['prediction'] == 1), 0)
    
    return tn, fp, fn, tp



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
#missing_count = df.filter(F.col("clan_location").isNull()).count()
#non_missing_count = df.count() - missing_count

#label = ['Missing value', 'Non-missing value']
#size = [missing_count, non_missing_count]
#color = ['lightblue', 'lightgreen']
#explode = (0.1, 0) # Explode the first slice for emphasis

#plt.pie(size, explode=explode, labels=label, colors=color, autopct='%1.1f%%', startangle=120)
#plt.title('Missing value in "clan_location" column')
#plt.axis('equal')
#plt.show()

# Clean df
df = clean_data(df)

# Check for null values in the entire dataframe
for column in df.columns:
    null_count = df.filter(F.col(column).isNull()).count()
    if null_count > 0:
        print(f"{column} has {null_count} null values")

# Impute or remove null values
df = df.na.drop()

# Store the cleaned df for later use
df.cache()
#df.count()



### Data preprocessing
# Convert srting cols to indexed numeric cols, excluding target cols
# Handle unseen labels using keep (debug point and take most of the time)


# Identify high cardinality col
threshold = 10000  # the problematic one which has more than 46000
high_cardinality_cols = []
for column in df.columns:
    if df.select(column).distinct().count() > threshold:
        high_cardinality_cols.append(column)

print("High cardinality columns:", high_cardinality_cols)

# Use a feature hasher for high-cardinality categorical features
hasher = FeatureHasher(inputCols=high_cardinality_cols, outputCol="hashed_features", numFeatures=1024)

# Exclude high cardinality columns from StringIndexer processing
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="skip").fit(df) 
            for column in list(set(df.columns)-set(['mean_member_trophies', 'mean_member_level']+high_cardinality_cols))]

# Combine the feature hasher and other indexers
stages = indexers + [hasher]

pipeline = Pipeline(stages=stages)
df_indexed = pipeline.fit(df).transform(df)

df_indexed.cache()

# Split data into training and testing set: 80% train, 20% test
# Set seed
seed = 202310
(train_data, test_data) = df_indexed.randomSplit([0.8, 0.2], seed=seed)

# Group by the clan type and calculate average member trophies and levels
df_grouped = train_data.groupBy("clan_type_index").agg({"mean_member_trophies": "avg", "mean_member_level": "avg"})

print(df_grouped.head(10))



### Modeling
# Feature columns
feature_columns = [col for col in df_indexed.columns if col not in ['clan_type', 'mean_member_trophies', 'mean_member_level']]
feature_columns.append("hashed_features")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid='skip')

# Feature scaling with standard scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)

# Logistic Regression to predict clan type
logistic_model = LogisticRegression(maxIter=20, labelCol="clan_type_index", featuresCol="features")
pipeline_logistic = Pipeline(stages=[assembler, scaler, logistic_model]) # Adding scaler into pipeline
# Train and predict
model = pipeline_logistic.fit(train_data)
predictions = model.transform(test_data)

# Random Forest model
random_forest_model = RandomForestClassifier(labelCol="clan_type_index", featuresCol="features", numTrees=100, maxBins=10000, maxDepth=10)
pipeline_rf = Pipeline(stages=[assembler, scaler, random_forest_model])
# Train and predict
model_rf = pipeline_rf.fit(train_data)
predictions_tree = model_rf.transform(test_data)

# Linear Regression to predict mean_member_trophies
linear_model = LinearRegression(maxIter=20, labelCol="mean_member_trophies", featuresCol="features")
pipeline_linear = Pipeline(stages=[assembler, scaler, linear_model])
# Train and predict
model_linear = pipeline_linear.fit(train_data)
predictions_linear = model_linear.transform(test_data)



### Model evaluation
# Evaluator for classification tasks
evaluator = MulticlassClassificationEvaluator(labelCol="clan_type_index")

# For Logistic Regression
accuracy_logistic = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1_logistic = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
recall_logistic = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

# For Random Forest
accuracy_tree = evaluator.evaluate(predictions_tree, {evaluator.metricName: "accuracy"})
f1_tree = evaluator.evaluate(predictions_tree, {evaluator.metricName: "f1"})
recall_tree = evaluator.evaluate(predictions_tree, {evaluator.metricName: "weightedRecall"})

print("----- Logistic Regression -----")
print(f"Accuracy: {accuracy_logistic}")
print(f"F1 Score: {f1_logistic}")
print(f"Recall: {recall_logistic}")

print("\n----- Random Forest -----")
print(f"Accuracy: {accuracy_tree}")
print(f"F1 Score: {f1_tree}")
print(f"Recall: {recall_tree}")

# Confusion matrix for Logistic regression
print("\nConfusion matrix for Logistic Regression:")
predictions.groupBy("clan_type_index", "prediction").count().show()

# Tn, fp, fn, tp
cm_logistic = predictions.groupBy("clan_type_index", "prediction").count().collect()
tn, fp, fn, tp = extract_confusion_metrics(cm_logistic)

print(f"Logistic Regression: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Confusion matrix for Random Forest
print("\nConfusion matrix for Random Forest:")
predictions_tree.groupBy("clan_type_index", "prediction").count().show()

confusion_matrix_rf = predictions_tree.groupBy("clan_type_index", "prediction").count().collect()
tn, fp, fn, tp = extract_confusion_metrics(confusion_matrix_rf)

print(f"Random Forest: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# For Linear Regression, we use RMSE as metrics
regression_evaluator = RegressionEvaluator(labelCol="mean_member_trophies", metricName="rmse")
rmse_linear = regression_evaluator.evaluate(predictions_linear)
print(f"\nRoot Mean Squared Error (RMSE) for Linear Regression: {rmse_linear}")



### Model optimization
# Logistic regression model with L2 regularization
logistic_l2 = LogisticRegression(featuresCol="features", labelCol="clan_type_index", maxIter=10, regParam=0.1, elasticNetParam=0)
pipeline_logistic_l2 = Pipeline(stages=[assembler, scaler, logistic_l2])
model_logistic_l2 = pipeline_logistic_l2.fit(train_data)
predictions_l2 = model_logistic_l2.transform(test_data)

# Metrics for Logistic Regression with L2 regularization
accuracy_logistic_l2 = evaluator.evaluate(predictions_l2, {evaluator.metricName: "accuracy"})
f1_logistic_l2 = evaluator.evaluate(predictions_l2, {evaluator.metricName: "f1"})
recall_logistic_l2 = evaluator.evaluate(predictions_l2, {evaluator.metricName: "weightedRecall"})

print("----- Logistic Regression with L2 Regularization -----")
print(f"Accuracy: {accuracy_logistic_l2}")
print(f"F1 Score: {f1_logistic_l2}")
print(f"Recall: {recall_logistic_l2}")

# Confusion matrix for Logistic Regression with L2
print("\nConfusion matrix for Logistic Regression with L2:")
predictions_l2.groupBy("clan_type_index", "prediction").count().show()

confusion_matrix_logistic_l2 = predictions_l2.groupBy("clan_type_index", "prediction").count().collect()
fn, fp, fn, tp = extract_confusion_metrics(confusion_matrix_logistic_l2)

print(f"Logistic Regression (L2): TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Optimized random forest model
rf_optimized = RandomForestClassifier(featuresCol="features", labelCol="clan_type_index", numTrees=150, maxDepth=10, maxBins=10000)
pipeline_rf_optimized = Pipeline(stages=[assembler, scaler, rf_optimized])
model_rf_optimized = pipeline_rf_optimized.fit(train_data)
predictions_rf_optimized = model_rf_optimized.transform(test_data)

# Metrics for optimized Random Forest
accuracy_rf_optimized = evaluator.evaluate(predictions_rf_optimized, {evaluator.metricName: "accuracy"})
f1_rf_optimized = evaluator.evaluate(predictions_rf_optimized, {evaluator.metricName: "f1"})
recall_rf_optimized = evaluator.evaluate(predictions_rf_optimized, {evaluator.metricName: "weightedRecall"})

print("\n----- Optimized Random Forest -----")
print(f"Accuracy: {accuracy_rf_optimized}")
print(f"F1 Score: {f1_rf_optimized}")
print(f"Recall: {recall_rf_optimized}")

# Confusion matrix for optimized Random Forest
print("\nConfusion matrix for Optimized Random Forest:")
predictions_rf_optimized.groupBy("clan_type_index", "prediction").count().show()

confusion_matrix_rf_optimized = predictions_rf_optimized.groupBy("clan_type_index", "prediction").count().collect()
tn, fp, fn, tp = extract_confusion_metrics(confusion_matrix_rf_optimized)

print(f"Optimized Random Forest: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# Define the optimized linear regression model
linear_optimized = LinearRegression(featuresCol="features", labelCol="mean_member_trophies", maxIter=10, regParam=0.1, elasticNetParam=0.2)
pipeline_linear_optimized = Pipeline(stages=[assembler, scaler, linear_optimized])
model_linear_optimized = pipeline_linear_optimized.fit(train_data)
predictions_linear_optimized = model_linear_optimized.transform(test_data)

# RMSE for optimized Linear Regression
rmse_linear_optimized = regression_evaluator.evaluate(predictions_linear_optimized)
print(f"\nRoot Mean Squared Error (RMSE) for Optimized Linear Regression: {rmse_linear_optimized}")





spark.stop()











import numpy as np
import time
import random

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, PCA
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler


spark = SparkSession.builder.appName("HeartDisease").getOrCreate()


def data_preprocessing(df):

    # Step 1: handling categorical values
    categorical_values = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]

    indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
                for col in categorical_values]

    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(df)
    data = model.transform(df)

    non_string_columns = [field.name for field in data.schema.fields if not isinstance(field.dataType, StringType)]
    data_numeric_only = data.select(*non_string_columns)

    # Step 2: assembling vector
    label_col = "HadHeartAttack_idx"
    feature_cols = [col for col in data_numeric_only.columns if col != label_col]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data_numeric_only = assembler.transform(data_numeric_only)

    return data_numeric_only


def raw_logistic_regression(prep_data):
    seed = random.randint(1, 9999)

    # Step 3: split data into train and test data
    train_data, test_data = prep_data.randomSplit([0.7, 0.3], seed=seed)

    # Step 4: create the classifier
    lr = LogisticRegression(featuresCol="features", labelCol="HadHeartAttack_idx")
    evaluator = MulticlassClassificationEvaluator(labelCol="HadHeartAttack_idx", predictionCol="prediction", metricName="accuracy")

    # Step 5: train the model
    start_time = time.time()
    model = lr.fit(train_data)
    end_time = time.time()

    # Step 6: test the model
    test_predictions = model.transform(test_data)
    train_predictions = model.transform(train_data)

    test_accuracy = evaluator.evaluate(test_predictions)
    train_accuracy = evaluator.evaluate(train_predictions)

    return train_accuracy, test_accuracy, end_time - start_time


def pca_regression(prep_data):
    k = 5
    pca = PCA(k=k, inputCol="features", outputCol="pca_features")

    pca_model = pca.fit(prep_data)
    pca_data = pca_model.transform(prep_data)

    pca_data = pca_data.select("pca_features", "HadHeartAttack_idx")
    pca_data = pca_data.withColumnRenamed("pca_features", "features")

    return pca_data


def normalised_logistic_regression(prep_data):

    # Step 3: normalise non-categorical data
    columns_to_normalise = [c for c in prep_data.columns[:-1] if not c.endswith("_idx")]

    for column in columns_to_normalise:
        q1, q99 = prep_data.approxQuantile(column, [0.01, 0.99], 0.25)
        prep_data = prep_data.withColumn(
                column,
                when(col(column) < q1, q1)
                .when(col(column) > q99, q99)
                .otherwise(col(column))
            )
    assembler = VectorAssembler(inputCols=columns_to_normalise, outputCol="features_raw")
    prep_data = assembler.transform(prep_data)

    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features_normalised")
    scaler_model = scaler.fit(prep_data)
    normalised_df = scaler_model.transform(prep_data)
    needed_columns = [field.name for field in normalised_df.schema.fields if field.name != "features"]
    needed_data = normalised_df.select(*needed_columns)

    # Step 4: assembling vector
    categorical_cols = [c for c in needed_data.columns if c.endswith("_idx") and c != "HadHeartAttack_idx"]
    final_assembler = VectorAssembler(
        inputCols=["features_normalised"] + categorical_cols,
        outputCol="features"
    )
    final_data = final_assembler.transform(needed_data)
    return final_data


def stats(inp_data):
    train_accuracies = []
    test_accuracies = []
    runtimes = []
    for i in range(10):
        train_accuracy, test_accuracy, runtime = raw_logistic_regression(inp_data)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        runtimes.append(runtime)
    print("Train accuracies: ")
    print(train_accuracies)
    print("Test accuracies: ")
    print(test_accuracies)
    print("Runtimes: ")
    print(runtimes)
    print("\n=== Summary of 10 Runs ===")
    print(f"Test Accuracy: min = {np.min(test_accuracies):.4f}, max = {np.max(test_accuracies):.4f}, "
          f"mean = {np.mean(test_accuracies):.4f}, std = {np.std(test_accuracies):.4f}")
    print(f"Train Accuracy: min = {np.min(train_accuracies):.4f}, max = {np.max(train_accuracies):.4f}, "
          f"mean = {np.mean(train_accuracies):.4f}, std = {np.std(train_accuracies):.4f}")
    print(f"Avg Runtime: {np.mean(runtimes):.4f} sec")


def main():
    df = spark.read.csv("/Users/mariapogorelova/AIML427/heart_disease_2022/heart_2022_no_nans.csv", header=True,
                        inferSchema=True)
    data = data_preprocessing(df)
    print("=== Raw Data ===")
    stats(data)
    print("=== Normalised Data ===")
    norm_data = normalised_logistic_regression(data)
    stats(norm_data)
    print("=== Normalised + PCA ===")
    norm_data = normalised_logistic_regression(data)
    pca_data = pca_regression(norm_data)
    stats(pca_data)


main()

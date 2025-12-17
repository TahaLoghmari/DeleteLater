from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from typing import List, Optional


def build_spark(app_name: str = "Alzheimer-Disease-Prediction", master: str = "local[*]") -> SparkSession:
    
    builder = SparkSession.builder.appName(app_name).master(master)
    builder = builder.config("spark.executor.memory", "2g").config("spark.driver.memory", "2g")
    return builder.getOrCreate()


def read_csv(spark: SparkSession, path: str, header: bool = True, infer_schema: bool = True) -> DataFrame:
    return spark.read.csv(path, header=header, inferSchema=infer_schema)


def assemble_features(df: DataFrame, input_cols: List[str], output_col: str = "features") -> Pipeline:
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    scaler = StandardScaler(inputCol=output_col, outputCol="scaledFeatures")
    return Pipeline(stages=[assembler, scaler])


def stratify_label(df: DataFrame, label_col: str) -> DataFrame:
    return df.withColumn(label_col, when(col(label_col) > 0, 1).otherwise(0))


def generate_dummy_data(path: str, rows: int = 100):

    import pandas as pd
    import numpy as np
    
    print(f"Generating dummy Alzheimer's disease data at {path}...")
    
    data = {
        "Age": np.random.randint(60, 95, rows),
        "Gender": np.random.randint(0, 2, rows),       # 0=female, 1=male
        "BMI": np.random.uniform(18.5, 40.0, rows),
        "Smoking": np.random.randint(0, 2, rows),
        "AlcoholConsumption": np.random.uniform(0, 20, rows),
        "PhysicalActivity": np.random.uniform(0, 10, rows),
        "DietQuality": np.random.uniform(0, 10, rows),
        "SleepQuality": np.random.uniform(4, 10, rows),
        "SystolicBP": np.random.randint(90, 180, rows),
        "DiastolicBP": np.random.randint(60, 120, rows),
        "CholesterolTotal": np.random.uniform(150, 300, rows),
        "MMSE": np.random.uniform(0, 30, rows),
        "Diagnosis": np.random.randint(0, 2, rows)     # 1=Alzheimer's, 0=Healthy
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print("Dummy Alzheimer's data generated.")

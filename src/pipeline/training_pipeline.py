# src/pipeline/training_pipeline.py

import os
import sys
from src.Exception import CustomException
from src.Logger import logging
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    A class to run the end-to-end training pipeline:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting Training Pipeline...")

            # Data Ingestion
            logging.info("Running Data Ingestion...")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Train path: {train_path}, Test path: {test_path}")

            # Data Transformation
            logging.info("Running Data Transformation...")
            train_array, test_array, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            # Model Training
            logging.info("Running Model Trainer...")
            r2_score_value = self.model_trainer.initiate_model_trainer(train_array, test_array)
            logging.info(f"Model training completed. Best model R2 score: {r2_score_value}")

            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    score = pipeline.run_pipeline()
    print(f"Training Pipeline executed successfully! Best model R2 score: {score}")

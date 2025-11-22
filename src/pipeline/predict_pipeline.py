import os
import sys
import pandas as pd
from src.Exception import CustomException
from src.Utils import load_object

class CustomData:
    """
    A class to structure user input data into a DataFrame
    for prediction.
    """
    def __init__(
        self,
        Item_Weight: float,
        Item_Visibility: float,
        Item_MRP: float,
        Outlet_Establishment_Year: int,
        Item_Fat_Content: str,
        Item_Type: str,
        Outlet_Size: str,
        Outlet_Location_Type: str,
        Outlet_Type: str
    ):
        self.Item_Weight = Item_Weight
        self.Item_Visibility = Item_Visibility
        self.Item_MRP = Item_MRP
        self.Outlet_Establishment_Year = Outlet_Establishment_Year
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Type = Item_Type
        self.Outlet_Size = Outlet_Size
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type = Outlet_Type

    def get_data_as_dataframe(self):
        """
        Convert input attributes to a pandas DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the input features
        """
        try:
            data_dict = {
                "Item_Weight": [self.Item_Weight],
                "Item_Visibility": [self.Item_Visibility],
                "Item_MRP": [self.Item_MRP],
                "Outlet_Establishment_Year": [self.Outlet_Establishment_Year],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Type": [self.Item_Type],
                "Outlet_Size": [self.Outlet_Size],
                "Outlet_Location_Type": [self.Outlet_Location_Type],
                "Outlet_Type": [self.Outlet_Type]
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    """
    A class to load the preprocessing object and trained model,
    and make predictions.
    """
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Predict the sales for the input features.
        
        Args:
            features (pd.DataFrame): DataFrame of input features
        
        Returns:
            np.ndarray: Array of predictions
        """
        try:
            # Load preprocessor and trained model
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Transform features using preprocessor
            features_transformed = preprocessor.transform(features)

            # Make predictions
            predictions = model.predict(features_transformed)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)

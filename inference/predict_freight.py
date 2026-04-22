import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "predict_freight.pkl")


def load_model(model_path:str=MODEL_PATH):

    with open(model_path,"rb") as f:
        model=joblib.load(f)

    return model



def predict_freight_cost(input_data):
    model=load_model()
    input_df=pd.DataFrame(input_data)
    input_df['Predict_Freight']=model.predict(input_df).round()
    return input_df

if __name__=="__main__":
    sample_data={
        "Dollars":[18000,9000,3000,200],
        "Quantity":[1000,34,200,3]
        }
    prediction=predict_freight_cost(sample_data)
    print(prediction)





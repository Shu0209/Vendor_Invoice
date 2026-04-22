import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "predict_flag_invoice.pkl")


def load_model(model_path:str=MODEL_PATH):

    with open(model_path,"rb") as f:
        model=joblib.load(f)

    return model



def predict_invoice_flag(input_data):
    model=load_model()
    input_df=pd.DataFrame(input_data)
    input_df['Predict_Flag']=model.predict(input_df).round()
    return input_df

if __name__=="__main__":
    sample_data={
        "invoice_quantity":[100,60,500,44],
        "invoice_dollars":[500,800,300,200],
        "Freight":[5,8,3,2],
        "total_item_quantity":[100,50,300,200],
        "total_item_dollars":[200,500,900,100],
        
        
        }
    prediction=predict_invoice_flag(sample_data)
    print(prediction)





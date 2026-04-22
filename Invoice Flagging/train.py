import joblib
from pathlib import Path

from data_preprocess import load_data,data_split,apply_lable
from model_evolution import Random_Forest,evaluate_model


features=['invoice_quantity',
          'invoice_dollars',
          'Freight',
          'total_item_quantity','total_item_dollars']
target="flag_invoice"

def main():
    db_path="../Dataset/inventory.db"
    

    df=load_data(db_path)
    df=apply_lable(df)

    
    x_train,x_test,y_train,y_test=data_split(df,features,target)


    grid_search=Random_Forest(x_train,y_train)

    evaluate_model(grid_search.best_estimator_,
                   x_test,
                   y_test,
                   "Random Forest")
    

    #Save Best Model
    joblib.dump(grid_search.best_estimator_,'../models/predict_flag_invoice.pkl')

if __name__=="__main__":
    main()



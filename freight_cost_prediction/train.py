import joblib
from pathlib import Path

from data_preprocess import load_data,feature_extraction,data_split
from model_evaluation import linear_reg,decision_tree,random_forest,evaluate_model

def main():
    db_path="../Dataset/inventory.db"
    model_dir=Path("models")
    model_dir.mkdir(exist_ok=True)


    df=load_data(db_path)
    X,Y=feature_extraction(df)
    x_train,x_test,y_train,y_test=data_split(X,Y)

    lr=linear_reg(x_train,y_train)
    dt=decision_tree(x_train,y_train)
    rf=random_forest(x_train,y_train)

    result=[]
    result.append(evaluate_model(lr,x_test,y_test,"Linear Regression"))
    result.append(evaluate_model(dt,x_test,y_test,"Decision Tree"))
    result.append(evaluate_model(rf,x_test,y_test,"Random Forest"))


    best_model_info=min(result,key=lambda x:x["mae"])
    best_model_name=best_model_info["model_name"]

    best_model={
        "Linear Regression":lr,
        "Decision Tree":dt,
        "Random Forest":rf
    }[best_model_name]

    model_path=".."/model_dir/"predict_freight.pkl"
    joblib.dump(best_model,model_path)

    print(f"\nBest Model Saved:{best_model_name}")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()



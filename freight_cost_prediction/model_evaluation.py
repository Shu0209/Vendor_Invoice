from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def linear_reg(x_train,y_train):
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    return lr

def decision_tree(x_train,y_train):
    dt=DecisionTreeRegressor(max_depth=3,random_state=42)
    dt.fit(x_train,y_train)
    return dt

def random_forest(x_train,y_train):
    rf=RandomForestRegressor(max_depth=5,random_state=42)
    rf.fit(x_train,y_train)
    return rf

def evaluate_model(model,x_test,y_test,model_name):
    pred=model.predict(x_test)
    mae=mean_absolute_error(pred,y_test)
    mse=mean_squared_error(pred,y_test)
    r2=r2_score(y_test,pred)*100

    # print(f"\n{model_name} peraformance:")
    # print(f"MAE: {mae:.2f}")
    # print(f"MSE: {mse:.2f}")
    # print(f"R2: {r2:.2f}%")

    return {
        "model_name":model_name,
        "mae":mae,
        "mse":mse,
        "r2":r2
    }



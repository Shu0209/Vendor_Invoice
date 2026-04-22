from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,f1_score,accuracy_score,precision_score,recall_score

def Random_Forest(x_train,y_train):
    rf=RandomForestClassifier(random_state=42,n_jobs=-1)

    param_grid={
        "n_estimators":[100,200,300],
        "max_depth":[None,3,5,7],
        "min_samples_split":[2,3,5],
        "min_samples_leaf":[1,2,5],
        "criterion":['gini','entropy']
        }
    score=make_scorer(f1_score)

    grid_search=GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=score,
        cv=5,
        verbose=5,
        n_jobs=-1
    )

    grid_search.fit(x_train,y_train)
    return grid_search


def evaluate_model(model,x_test,y_test,model_name):
    pred=model.predict(x_test)
    precision=precision_score(y_test,pred)
    f1=f1_score(y_test,pred)
    recall=recall_score(y_test,pred)
    accuracy=accuracy_score(y_test,pred)*100

    print(f"\n{model_name} peraformance:")
    print(f"Precision: {precision:.2f}")
    print(f"F1_Score: {f1:.2f}")
    print(f"Recall: {recall:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")

    return {
        "model_name":model_name,
        "precision":precision,
        "f1_score":f1,
        "Recall_Score":recall,
        "Accuracy":accuracy
    }

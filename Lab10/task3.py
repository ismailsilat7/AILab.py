import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TEST_SIZE = 0.4

FEATURE_NAMES = ["Monthly Charges","Contract Type","Tenure (months)","Internet Service Type","Customer Support Calls"]

def main():
    if len(sys.argv)!=2:
        sys.exit("Usage: python task3.py data.csv")
    evidence,labels=load_data(sys.argv[1])
    if not evidence:
        sys.exit("No valid data loaded.")
    print(f"Loaded {len(evidence)} valid records.\n")

    evidence=treat_outliers(evidence,FEATURE_NAMES)
    scaler=StandardScaler()
    evidence_scaled=scaler.fit_transform(evidence)
    X_train,X_test,y_train,y_test=train_test_split(evidence_scaled,labels,test_size=TEST_SIZE,random_state=42)
    model=train_model(X_train,y_train)

    print("Feature Importance")
    feature_importance(model,FEATURE_NAMES)
    print()

    print("Decision Boundary")
    print("Random Forest uses an ensemble of decision trees.")
    print("Each tree splits on feature thresholds — majority vote forms boundary.")
    print()

    predictions=model.predict(X_test)

    print("Confusion Matrix")
    print_confusion_matrix(y_test,predictions)
    print()

    accuracy,precision,recall,f1=evaluate(y_test,predictions)
    print("Model Evaluation")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print()

    new_customer=[[85.0,0,5,2,6]]
    new_customer_scaled=scaler.transform(new_customer)
    pred=model.predict(new_customer_scaled)
    print("New Customer Prediction")
    print("Will Churn!" if pred[0]==1 else "Will Stay.")
    

def load_data(filename):
    evidence,labels=[],[]
    contract_map={"Month-to-Month":0,"One Year":1,"Two Year":2}
    internet_map={"None":0,"DSL":1,"Fiber Optic":2}
    with open(filename) as f:
        reader=csv.reader(f)
        next(reader)
        for row in reader:
            try:
                m=float(row[1]) if row[1].strip() else None
                c=contract_map.get(row[2].strip(),None)
                t=float(row[3]) if row[3].strip() else None
                i=internet_map.get(row[4].strip(),None)
                s=float(row[5]) if row[5].strip() else None
                y=int(row[6]) if row[6].strip() else None
                if None in [m,c,t,i,s,y]:
                    continue
                evidence.append([m,c,t,i,s])
                labels.append(y)
            except:
                continue
    return evidence,labels

def treat_outliers(evidence,feature_names):
    treated=[row[:] for row in evidence]
    for col in range(len(treated[0])):
        values=sorted(row[col] for row in treated)
        n=len(values)
        q1=values[n//4]
        q3=values[(3*n)//4]
        iqr=q3-q1
        low=q1-1.5*iqr
        high=q3+1.5*iqr
        for row in treated:
            row[col]=max(low,min(high,row[col]))
    return treated

def train_model(evidence,labels):
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(evidence,labels)
    return model

def feature_importance(model,feature_names):
    imp=list(zip(feature_names,model.feature_importances_))
    imp.sort(key=lambda x:x[1],reverse=True)
    for n,s in imp:
        print(n,round(s,4))

def print_confusion_matrix(labels,predictions):
    cm=confusion_matrix(labels,predictions)
    tn,fp,fn,tp=cm.ravel()
    print(tn,fp)
    print(fn,tp)

def evaluate(labels,predictions):
    return (
        accuracy_score(labels,predictions),
        precision_score(labels,predictions,zero_division=0),
        recall_score(labels,predictions,zero_division=0),
        f1_score(labels,predictions,zero_division=0)
    )

if __name__=="__main__":
    main()
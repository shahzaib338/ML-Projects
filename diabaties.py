import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = ['pregnancies', 'glucose', 'blood_pressure', 
           'skin_thickness', 'insulin', 'bmi', 
           'diabetes_pedigree', 'age', 'outcome']

df = pd.read_csv(url, names=columns)

col_to_fix = ["glucose", "blood_pressure", "bmi"]
df[col_to_fix] = df[col_to_fix].replace(0, np.nan)


df["glucose"] = df['glucose'].fillna(df['glucose'].median())
df["blood_pressure"] = df['blood_pressure'].fillna(df['blood_pressure'].median())
df["bmi"] =  df['bmi'].fillna(df['bmi'].median())

df = df.drop("skin_thickness", axis=1)
df = df.drop("insulin", axis=1)

print(df.shape)
print(df.head())
print(df.isnull().sum())

X = df.drop("outcome", axis=1)
y = df["outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name} has {acc} accuracy")

best_model = models["Random Forest Classifier"]
while True:
    try:
        pregnancies = int(input("Enter pregnancies: "))
        glucose = float(input("Enter glucose: "))
        blood_pressure = float(input("Enter Blood Pressure: "))
        bmi = float(input("Enter bmi: "))
        diabetes_pedigree = float(input("Enter diabetes_pedigree: "))
        age = int(input("Enter AGE: "))

        new_patient = np.array([[pregnancies, glucose,
                                  blood_pressure, bmi,
                                  diabetes_pedigree, age]])
        new_patient_scaled = scaler.transform(new_patient)
        prediction = best_model.predict(new_patient_scaled)
        print("Diabetes Detected! 🔴" if prediction[0] == 1 else "No Diabetes! 🟢")

        again = input("\nDo you want to check another patient? (yes/no): ")
        if again.lower() == 'yes':
            continue
        else:
            print("Goodbye! Stay Healthy! 🟢")
            break

    except ValueError:
        print("❌ Please enter valid numbers! Try again!\n")
    except Exception as e:
        print("❌ Error:", e)
        break
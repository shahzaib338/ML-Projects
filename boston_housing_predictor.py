import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)

print(df.head())

scaler = StandardScaler()
X = scaler.fit_transform(df)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
    "XGBRegressor": XGBRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae =  mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"\n{name} Mae is {mae} MSE is {mse} and R2 is {r2}")
    
    best_model = models["XGBRegressor"]
while True:
    try:
        MedInc = float(input("Enter MedInc: "))
        HouseAge = float(input("Enter House Age: "))
        AveRooms = float(input("Enter AveRooms: "))
        AveBedrms = float(input("Enter Avebedrms: "))
        Population = float(input("Enter Population: "))
        AveOccup = float(input("Enter AveOccup: "))
        Latitude = float(input("Enter Latitude: "))
        Longitude = float(input("Enter Longitude: "))

        cols = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
        new_price = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,Longitude]], columns=cols)
        new_price_scaled = scaler.transform(new_price)
        Price_prd = best_model.predict(new_price_scaled)
        print(f"Predicted House Price: ${Price_prd[0]*100000:.2f}")

        again = input("\nCheck another house? (yes/no): ")
        if again.lower() != 'yes':
            print("Goodbye! 🏠")
            break


    except ValueError:
        print("Please Enter a valid Value")
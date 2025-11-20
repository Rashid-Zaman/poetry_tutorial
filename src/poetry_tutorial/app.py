from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import pathlib

app = Flask(__name__)
BASE_DIR = pathlib.Path(__file__).resolve().parent
ridge_model=pickle.load(open(BASE_DIR/'ridge.pkl','rb'))
standard_scaler=pickle.load(open(BASE_DIR/'scaler.pkl','rb'))
ohe=pickle.load(open(BASE_DIR/'ohe.pkl','rb'))

# Feature names (excluding the target)
feature_names = [
    "Distance_km", "Weather", "Traffic_Level", "Time_of_Day",
    "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"
]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/zamanrashid_features')
def features():
    return render_template("features.html", features=feature_names)

@app.route('/zamanrashid_predict', methods=["POST"])
def predict():
    try:
        input_data = [request.form.get(feat) for feat in feature_names]
        print(input_data)
        # Convert to correct types
        input_data = pd.DataFrame([input_data], columns=feature_names)
        input_data["Distance_km"] = input_data["Distance_km"].astype(float)
        input_data["Preparation_Time_min"] = input_data["Preparation_Time_min"].astype(float)
        input_data["Courier_Experience_yrs"] = input_data["Courier_Experience_yrs"].astype(float)
        encoded=ohe.transform(input_data[["Weather", "Traffic_Level", "Time_of_Day","Vehicle_Type"]])
        print(encoded)
        encoded_df=pd.DataFrame(encoded,columns=ohe.get_feature_names_out())
        input_data.drop(columns=["Weather", "Traffic_Level", "Time_of_Day","Vehicle_Type"],axis=1,inplace=True)
        print(input_data.head(2))
        total_df=pd.concat([input_data,encoded_df],axis=1)
        print(total_df.head())
        new_data=np.array(total_df)
        new_data_scaled=standard_scaler.transform(new_data)
        print(new_data_scaled)
        result=ridge_model.predict(new_data_scaled)
        return render_template('result.html',results=result[0])
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
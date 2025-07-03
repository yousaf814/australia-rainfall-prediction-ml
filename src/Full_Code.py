!pip install scikit-learn --upgrade --quiet
!pip install pandas --quiet
!pip install plotly matplotlib seaborn --quiet
!pip install numpy --quiet
!pip install -q streamlit
!pip install streamlit
!npm install localtunnel


%%writefile Australia_Weather_Predictor_app.py

import opendatasets as od
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Download the dataset
raw_df = pd.read_csv('/content/weatherAUS.csv')
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Create training, validation and test sets
year = pd.to_datetime(raw_df.Date).dt.year
train_df, val_df, test_df = raw_df[year < 2015], raw_df[year == 2015], raw_df[year > 2015]

# Create inputs and targets
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
test_inputs, test_targets = test_df[input_cols].copy(), test_df[target_col].copy()

# Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()[:-1]

# Impute missing numerical values
imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scale numeric features
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# Save processed data to disk
train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')
pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

# Load processed data from disk
train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')
train_targets = pd.read_parquet('train_targets.parquet')[target_col]
val_targets = pd.read_parquet('val_targets.parquet')[target_col]
test_targets = pd.read_parquet('test_targets.parquet')[target_col]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Select the columns to be used for training/prediction
X_train = train_inputs[numeric_cols]
X_val = val_inputs[numeric_cols]
X_test = test_inputs[numeric_cols]

# Create and train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, train_targets)

# Generate predictions and probabilities
train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
accuracy_score(train_targets, train_preds)

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    X_input = input_df[numeric_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

import streamlit as st

st.title('Australia Weather Predictor')
st.write("This app predicts Austraila Weather Rain")
st.write("The following is the dataset")
st.dataframe(raw_df)
st.write("Fill the folloing details to predict")
MinTemp = st.number_input('MinTemp')
MaxTemp = st.number_input('MaxTemp')
Rainfall = st.number_input('Rainfall')
Evaporation = st.number_input('Evaporation')
Sunshine = st.number_input('Sunshine')
WindGustSpeed = st.number_input('WindGustSpeed')
WindSpeed9am = st.number_input('WindSpeed9am')
WindSpeed3pm = st.number_input('WindSpeed3pm')
Humidity9am = st.number_input('Humidity9am')
Humidity3pm = st.number_input('Humidity3pm')
Pressure9am = st.number_input('Pressure9am')
Pressure3pm = st.number_input('Pressure3pm')
Cloud9am = st.number_input('Cloud9am')
Cloud3pm = st.number_input('Cloud3pm')
Temp9am = st.number_input('Temp9am')
Temp3pm = st.number_input('Temp3pm')

user_input = {'MinTemp': MinTemp, 'MaxTemp': MaxTemp, 'Rainfall': Rainfall, 'Evaporation': Evaporation, 'Sunshine': Sunshine, 'WindGustSpeed': WindGustSpeed, 'WindSpeed9am': WindSpeed9am, 'WindSpeed3pm': WindSpeed3pm, 'Humidity9am': Humidity9am, 'Humidity3pm': Humidity3pm, 'Pressure9am': Pressure9am, 'Pressure3pm': Pressure3pm, 'Cloud9am': Cloud9am, 'Cloud3pm': Cloud3pm, 'Temp9am': Temp9am, 'Temp3pm': Temp3pm}

Predict = predict_input(user_input)
if st.button('Predict'):
          st.write(Predict)


# Now runing the streamlit application by telling the name of the file "audio_to_text_text_to_generated_audio.py".

!streamlit run Australia_Weather_Predictor_app.py &>/content/logs.txt &

!wget -q -O - ipv4.icanhazip.com

!npx localtunnel --port 8501

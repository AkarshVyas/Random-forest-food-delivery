import streamlit as st
import pickle
import pandas as pd

st.title('Food Delivery Time Prediction')

# Load the optimized Random Forest model
try:
    with open('optimized_rf_model.pkl', 'rb') as file:
        optimized_rf_model = pickle.load(file)
    st.success('Optimized Random Forest model loaded successfully!')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the LabelEncoders
try:
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    st.success('LabelEncoders loaded successfully!')
except Exception as e:
    st.error(f"Error loading LabelEncoders: {e}")
    st.stop()

# Define the input fields based on the features used in training
# ['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs']

st.header('Enter Delivery Details:')

distance_km = st.number_input('Distance (km)', min_value=0.1, max_value=100.0, value=10.0, step=0.1)
preparation_time_min = st.number_input('Preparation Time (min)', min_value=1, max_value=60, value=20)
courier_experience_yrs = st.number_input('Courier Experience (years)', min_value=0.0, max_value=20.0, value=2.0, step=0.1)

# For categorical features, use selectbox and then transform with LabelEncoder
weather_options = label_encoders['Weather'].classes_
weather_selected = st.selectbox('Weather', weather_options)

traffic_level_options = label_encoders['Traffic_Level'].classes_
traffic_level_selected = st.selectbox('Traffic Level', traffic_level_options)

time_of_day_options = label_encoders['Time_of_Day'].classes_
time_of_day_selected = st.selectbox('Time of Day', time_of_day_options)

vehicle_type_options = label_encoders['Vehicle_Type'].classes_
vehicle_type_selected = st.selectbox('Vehicle Type', vehicle_type_options)


if st.button('Predict Delivery Time'):
    # Preprocess categorical inputs using the loaded LabelEncoders
    weather_encoded = label_encoders['Weather'].transform([weather_selected])[0]
    traffic_level_encoded = label_encoders['Traffic_Level'].transform([traffic_level_selected])[0]
    time_of_day_encoded = label_encoders['Time_of_Day'].transform([time_of_day_selected])[0]
    vehicle_type_encoded = label_encoders['Vehicle_Type'].transform([vehicle_type_selected])[0]

    # Create a DataFrame for prediction, ensuring correct column order and names
    input_data = pd.DataFrame([[distance_km, weather_encoded, traffic_level_encoded, time_of_day_encoded,
                                  vehicle_type_encoded, preparation_time_min, courier_experience_yrs]],
                                columns=['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day',
                                           'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs'])

    # Make prediction
    prediction = optimized_rf_model.predict(input_data)

    st.subheader(f'Predicted Delivery Time: {prediction[0]:.2f} minutes')

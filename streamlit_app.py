import numpy as np
import pickle
import streamlit as st

# Load the saved models
loaded_model1 = pickle.load(open("Heating load_model.sav", 'rb'))
loaded_model2 = pickle.load(open("Cooling load_model.sav", 'rb'))

def building_load_prediction(input_data):
    try:
        # Convert input_data to float and reshape
        input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    
        HeatingLoad = loaded_model1.predict(input_data_as_numpy_array)
        CoolingLoad = loaded_model2.predict(input_data_as_numpy_array)
    
        return HeatingLoad[0], CoolingLoad[0], None  # Return results and no error message
    except Exception as e:
        return None, None, str(e)  # Return None for results and the error message

def validate_input(input_data):
    error_messages = []

    for val in input_data:
        try:
            # Attempt to convert input data to float
            val = float(val)

            # Check for negative values
            if val < 0:
                error_messages.append(f"Value {val} cannot be negative.")
        except ValueError as ve:
            error_messages.append(f"Invalid input value: {ve}")

    if not all(input_data):
        error_messages.append("All fields must be filled in.")

    return input_data, error_messages

def main():
    st.title('Energy Efficient Building Load Calculation Web App')
    
    # Collect input data from the user
    
    # Getting the input data from the user
    col1, col2 = st.columns(2)
    
    with col1:
        RelativeCompactness = st.text_input('1. Relative compactness value')
    
    with col2:
        SurfaceArea = st.text_input('2. Surface Area value')
    
    with col1:
        WallArea = st.text_input('3. Wall Area value')
    
    with col2:
        RoofArea = st.text_input('4. Roof Area value')
    
    with col1:
        OverallHeight = st.text_input('5. Overall Height value')
    
    with col2:
        Orientation = st.text_input('6. Orientation value')
    
    with col1:
        GlazingArea = st.text_input('7. Glazing Area value')
    
    with col2:
        GlazingAreaDistribution = st.text_input('8. Glazing Area Distribution value')
    
    # Create placeholders for displaying the results and error messages
    HeatingLoad_result = ""
    CoolingLoad_result = ""
    error_messages = []
    
    # Check if the user has clicked the prediction button
    if st.button('Estimate the Load'):
        input_data = [RelativeCompactness, SurfaceArea, WallArea, RoofArea, OverallHeight, Orientation, GlazingArea, GlazingAreaDistribution]
        
        # Validate input data
        input_data, validation_errors = validate_input(input_data)
        
        if validation_errors:
            error_messages.extend(validation_errors)
        else:
            heating_load, cooling_load, prediction_error = building_load_prediction(input_data)
            
            if prediction_error is not None:
                error_messages.append(f"Error during prediction: {prediction_error}")
            else:
                HeatingLoad_result = f"Estimated Heating Load value is: {round(heating_load, 2)}"  # Display heating load
                CoolingLoad_result = f"Estimated Cooling Load value is: {round(cooling_load, 2)}"  # Display cooling load
    
    # Display the results or error messages in the Streamlit app
    if error_messages:
        for error in error_messages:
            st.error(error)
    else:
        st.success(HeatingLoad_result)
        st.success(CoolingLoad_result)

if __name__ == '__main__':
    main()

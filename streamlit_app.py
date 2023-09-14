import numpy as np
import pickle
import streamlit as st

# Load the saved models
loaded_model1 = pickle.load(open("D:/Datascience/Data Science Project Internships/iNeuron/Energy Efficiency Project/ENB SSM/Heating load_model.sav", 'rb'))
loaded_model2 = pickle.load(open("D:/Datascience/Data Science Project Internships/iNeuron/Energy Efficiency Project/ENB SSM/Cooling load_model.sav", 'rb'))

def building_load_prediction(input_data):
    # Convert input_data to float and reshape
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    
    HeatingLoad = loaded_model1.predict(input_data_as_numpy_array)
    CoolingLoad = loaded_model2.predict(input_data_as_numpy_array)
    
    return HeatingLoad, CoolingLoad

def main():
    st.title('Energy Efficient Building Load Calculation Web App')
    
    # Collect input data from the user
    RelativeCompactness = st.text_input('Relative compactness value')
    SurfaceArea = st.text_input('Surface Area value')
    WallArea = st.text_input('Wall Area value')
    RoofArea = st.text_input('Roof Area value')
    OverallHeight = st.text_input('Overall Height value')
    Orientation = st.text_input('Orientation value')
    GlazingArea = st.text_input('Glazing Area value')
    GlazingAreaDistribution = st.text_input('Glazing Area Distribution value')
    
    # Create placeholders for displaying the results
    HeatingLoad_result = ""
    CoolingLoad_result = ""
    
    # Check if the user has clicked the prediction button
    if st.button('Calculate Building Load'):
        input_data = [RelativeCompactness, SurfaceArea, WallArea, RoofArea, OverallHeight, Orientation, GlazingArea, GlazingAreaDistribution]
        heating_load, cooling_load = building_load_prediction(input_data)
        
        HeatingLoad_result = f"Calculated Heating Load value is: {heating_load[0]}"  # Display heating load
        CoolingLoad_result = f"Calculated Cooling Load value is: {cooling_load[0]}"  # Display cooling load
    
    # Display the results in the Streamlit app
    st.success(HeatingLoad_result)
    st.success(CoolingLoad_result)

if __name__ == '__main__':
    main()

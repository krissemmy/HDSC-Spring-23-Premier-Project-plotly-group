

import pickle
import streamlit as st

# Load the trained model
with open('C:/Users/pc/Downloads/hamoye_group_project_real/model (1).pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    st.title("Chronic Disease Prediction")

    # Create input form for patient information
age = st.number_input('Age', min_value=1, max_value=120, value=30)
bp = st.number_input('Blood Pressure (mmHg)', min_value=0, value=80)
sg = st.number_input('Specific Gravity', min_value=0.0, max_value=2.0, value=1.010)
al = st.number_input('Albumin', min_value=0, value=0)
su = st.number_input('Sugar', min_value=0, value=0)
rbc = st.selectbox('Red Blood Cells', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
pc = st.selectbox('Pus Cell', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
pcc = st.selectbox('Pus Cell Clumps', [0, 1], format_func=lambda x: 'Not Present' if x == 0 else 'Present')
ba = st.selectbox('Bacteria', [0, 1], format_func=lambda x: 'Not Present' if x == 0 else 'Present')
bgr = st.number_input('Random Blood Glucose (mg/dL)', min_value=0, value=100)
bu = st.number_input('Blood Urea', min_value=0, value=50)
sc = st.number_input('Serum Creatinine', min_value=0.0, value=1.2)
sod = st.number_input('Sodium (mEq/L)', min_value=0, value=135)
pot = st.number_input('Potassium (mEq/L)', min_value=0.0, value=4.0)
hemo = st.number_input('Hemoglobin (g/dL)', min_value=0.0, value=15.0)
pcv = st.number_input('Packed Cell Volume (%)', min_value=0, value=40)
wbcc = st.number_input('White Blood Cell Count (cells/cubic mm)', min_value=0, value=8000)
rbcc = st.number_input('Red Blood Cell Count (millions/cubic mm)', min_value=0, value=5)
htn = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
dm = st.selectbox('Diabetes Mellitus', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
cad = st.selectbox('Coronary Artery Disease', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
appet = st.selectbox('Appetite', [0, 1], format_func=lambda x: 'Poor' if x == 0 else 'Good')
pe = st.selectbox('Pedal Edema', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
ane = st.selectbox('Anemia', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
mcv = st.number_input('Mean Corpuscular Volume (MCV)', min_value=0, value=80)
glucose_bp_ratio = st.number_input('Glucose-to-Blood Pressure Ratio', min_value=0, value=0)
total_blood_cell_count = st.number_input('Total Blood Cell Count', min_value=0, value=0)
anemia = st.selectbox('Anemia Indicator', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
bp_category = st.selectbox('Blood Pressure Category', [0, 1, 2], format_func=lambda x: 'Low' if x == 0 else ('Normal' if x == 1 else 'High'))

if st.button('Predict'):
    # Make predictions
    input_data = [[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                   sc, sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad,
                   appet, pe, ane, mcv, glucose_bp_ratio,
                   total_blood_cell_count, anemia, bp_category]]


        # Convert input data to DataFrame and make predictions
    prediction = model.predict(input_data)
        
    # Display prediction
    if prediction[0] == 0:
        st.write('Prediction: CKD (Chronic Kidney Disease)')
    else:
        st.write('Prediction: No CKD (No Chronic Kidney Disease)')

# Run the app
if __name__ == '__main__':
    main()
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data= pd.read_csv("../data/data.csv")
    data = data.drop(["Unnamed: 32","id"], axis=1)
    data['diagnosis'] = data['diagnosis'].map({"M":1, "B":0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei measurements")

    data= get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dictionary={} #to create key value to pairs to actuallt store the slider info and use it to predict

    for label, key in slider_labels:
        input_dictionary[key] = st.sidebar.slider(
            label, 
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()), #values that are set by default
        )

    return input_dictionary

def get_scaled_values(input_dict):
    data=get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    #we can also use scler from scikit-learn
    for key, value in input_dict.items():
        max_val=X[key].max()
        min_val=X[key].min()
        scaled_val=(value-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_val

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
  
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                    'Smoothness', 'Compactness', 
                    'Concavity', 'Concave Points',
                    'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]    #this is the range of values in the graph, all the values are plotted here, hence the values have to be normalised(converted to 0-1) and then plot it to give a better representing graph
        )),
        showlegend=True
    )
    
    return fig

def add_predictions(input_data):

    model=pickle.load(open("../model/model.pkl", "rb"))
    scaler=pickle.load(open("../model/scaler.pkl", "rb"))

    #we are converting all the data from the sliders into a single array with values

    input_array = np.array(list(input_data.values())).reshape(1,-1) #the reshape() changes it multiple rows, one column to one row, multiple columns (from the array getting displayed vertically to horizontally)
    
    input_array_scaled=scaler.transform(input_array)    #if the input value is below the mean value, the scaled value goes negative and if the input is above the mean, then the sclaed one becomes positive, and if the input is equal to the mean value, then the scaled value is 0 
    
    prediction= model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0]==0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

    # st.markdown("""
    #     <style>
    #     .css-1l03f6z {
    #         background-color: lightgray !important;
    #     }
    #     </style>
    # """, unsafe_allow_html=True)
    # st.write(input_array)
    # st.write(input_array_scaled)
    # st.write(prediction)

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female doctor: ",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    #streamlit does not allow css to be added
    #this is a workaround which imports css file as a markdown file and allows the parsing of html
    with open("../assests/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()
    # st.write(input_data)    #displays the input values that are changed by the user through sliders which we will be using to make predictions

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
  
    col1, col2 = st.columns([3,1])  #first column is going to be 4 times as big as the second column

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()

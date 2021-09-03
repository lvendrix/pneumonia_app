import streamlit as st
from PIL import Image
from functions import x_ray_classification, pneumonia_classification

# Title of the page
st.set_page_config(
        page_title="Pneumonia App",
)

# Title and introduction
st.title("🫁 Pneumonia Identifier App")
st.write("""
This app detects whether or not lungs are infected by pneumonia, based on a X Ray image.
\nBuilt using Keras (Tensorflow).
\nThe original dataset can be found on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). 
Download a sample of lungs X Ray images [here](https://www.dropbox.com/sh/te71i0lli3qf6vc/AAA7otvsBNnY81L97JNv7znWa?dl=0).
\nThis app is __NOT__ meant for medical purposes!
""")

# file upload and handling logic
uploaded_file = st.file_uploader("Upload a X Ray image (.JPEG) of lungs", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    st.write("")
    st.subheader("Result")

    # Load models
    label_x_ray = x_ray_classification(image, 'keras_model_x_ray.h5')
    label_pneumonia = pneumonia_classification(image, 'keras_model_pneumonia.h5')

    # if image is not a X-Ray image
    if label_x_ray == 0:
        st.write("""
                This doesn't look like a X Ray image of lungs.
                \nPlease restart with a proper image.
                """)
    # x ray image
    else:
        # normal lungs
        if label_pneumonia == 0:
            st.write("This X Ray looks normal. This X Ray depicts __CLEAR LUNGS__.")
        # infected lungs
        else:
            st.write("""
            ⚠️ The lungs seem to be infected by __PNEUMONIA__.
            \nLearn more about pneumonia [here](https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204).
            \nFor further investigation, please contact your doctor.""")

# Cleans Streamlit layout
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
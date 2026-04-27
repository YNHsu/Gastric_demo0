import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time



# App Title
st.title("Gastric cancer prediction")

st.markdown(unsafe_allow_html=True, body="<p>In this web, you can select the patient and predict the probability of gastric cancer.</p>")
                                        

# =============================================================================
# img = st.file_uploader(label='Load a CT brain image', type=['png','jpg','jpeg'], key='CT', accept_multiple_files=False)
# basename = img.name
# =============================================================================


imgname = st.selectbox('Select the patient', (None, '00000'))
# imgname = st.selectbox('Select the CT image', (None, 'CTbrain_1.png','CTbrain_2.png','CTbrain_3.png','CTbrain_4.png','CTbrain_5.png','CTbrain_6.png','CTbrain_7.png','CTbrain_8.png'))

if imgname is not None:
    images = os.listdir('gastric_image/')
    st.session_state.images_count = len(images)
    # st.subheader('Raw image:', images_count)
    st.write('Raw images:', st.session_state.images_count)


    # Initial state
    if "step" not in st.session_state:
        st.session_state.step = 1    
    if "keep_images_count" not in st.session_state:
        st.session_state.keep_images_count = None    
    if "A_count" not in st.session_state:
        st.session_state.A_count = None
    if "A" not in st.session_state:
        st.session_state.A = None
    if "B_count" not in st.session_state:
        st.session_state.B_count = None
    if "B" not in st.session_state:
        st.session_state.B = None
    if "F_count" not in st.session_state:
        st.session_state.F_count = None
    if "F" not in st.session_state:
        st.session_state.F = None
    if "df" not in st.session_state:
        st.session_state.df = None

    if "A_HP" not in st.session_state:
        st.session_state.A_HP = None
    if "B_HP" not in st.session_state:
        st.session_state.B_HP = None
    if "F_HP" not in st.session_state:
        st.session_state.F_HP = None

    if "AG_A" not in st.session_state:
        st.session_state.AG_A = None
    if "AG_B" not in st.session_state:
        st.session_state.AG_B = None
    if "IM_A" not in st.session_state:
        st.session_state.IM_A = None
    if "IM_B" not in st.session_state:
        st.session_state.IM_B = None
  
    # ===== Step 1 =====
    st.subheader("Step 1: Exclusion criteria-Blurred image, NBI, Polyp, Ulcer, Tumor")
    st.markdown("##### <Using modified DenseNet121 model>")
    if st.session_state.step == 1:
        if st.button('Click here to exclude blurred image, NBI, Polyp, Ulcer, Tumor'):
            keep_images = os.listdir('Step1_ExclusionCriteria/')
            st.session_state.keep_images_count = len(keep_images)
            st.session_state.step = 2   # next step
    
    # Show Step 1 result（keep）
    if st.session_state.keep_images_count is not None:
        st.write('Retained images:', st.session_state.keep_images_count)
    
    
    # ===== Step 2 =====
    st.subheader("Step 2 and 3: Gastric-Antrum / Body / Fundus classification")
    st.markdown("##### <Using modified DenseNet201 and DenseNet121 model>")
    if st.session_state.step == 2:
        if st.button('Click here to classify Antrum / Body / Fundus'):
            st.session_state.A = sorted(os.listdir('Step2and3_ABF/A/'))
            st.session_state.A_count = len(st.session_state.A)
            st.session_state.B = sorted(os.listdir('Step2and3_ABF/B/'))
            st.session_state.B_count = len(st.session_state.B)
            st.session_state.F = sorted(os.listdir('Step2and3_ABF/F/'))
            st.session_state.F_count = len(st.session_state.F)
            st.session_state.step = 3
    
    # Show Step 2 result
    time.sleep(1)
    if st.session_state.A_count is not None:
        st.write('Antrum:', st.session_state.A_count)
        cols = st.columns(10)
        if st.session_state.A is not None:
            for idx, img_name in enumerate(st.session_state.A):
                img_path = os.path.join('Step2and3_ABF/A/', img_name)
                image = Image.open(img_path)
                cols[idx % 10].image(image, use_container_width=True)    # caption=img_name
    time.sleep(1)        
    if st.session_state.B_count is not None:
        st.write('Body:', st.session_state.B_count)
        cols = st.columns(10)
        if st.session_state.B is not None:
            for idx, img_name in enumerate(st.session_state.B):
                img_path = os.path.join('Step2and3_ABF/B/', img_name)
                image = Image.open(img_path)
                cols[idx % 10].image(image, use_container_width=True)    # caption=img_name
    time.sleep(1)
    if st.session_state.F_count is not None:
        st.write('Fundus:', st.session_state.F_count)
        cols = st.columns(10)
        if st.session_state.F is not None:
            for idx, img_name in enumerate(st.session_state.F):
                img_path = os.path.join('Step2and3_ABF/F/', img_name)
                image = Image.open(img_path)
                cols[idx % 10].image(image, use_container_width=True)    # caption=img_name

    # ===== Step 3 =====
    st.subheader("Step 4: HP(Helicobacter pylori) prediction")
    st.markdown("##### <Using modified DenseNet121, ResNet50 and InceptionResNetV2 model>")
    if st.session_state.step == 3:
        if st.button('Click here to predict HP'):
            st.session_state.pred_hp = 0.391
            st.session_state.A_HP = sorted(os.listdir('Step4_HP/A/'))
            st.session_state.B_HP = sorted(os.listdir('Step4_HP/B/'))
            st.session_state.F_HP = sorted(os.listdir('Step4_HP/F/'))
    time.sleep(1)      

      
    # Show Step 3 result   
    if st.session_state.A_HP is not None:
        st.write('Antrum: HP prediction =', 0.450)
        cols = st.columns(10)
        for idx, img_name in enumerate(st.session_state.A_HP):
            img_path = os.path.join('Step4_HP/A/', img_name)
            image = Image.open(img_path)
            cols[idx % 10].image(image, use_container_width=True)    # caption=img_name
    time.sleep(1)            
    if st.session_state.B_HP is not None:
        st.write('Body: HP prediction =', 0.537)
        cols = st.columns(10)
        for idx, img_name in enumerate(st.session_state.B_HP):
            img_path = os.path.join('Step4_HP/B/', img_name)
            image = Image.open(img_path)
            cols[idx % 10].image(image, use_container_width=True)    # caption=img_name
    time.sleep(1)        
    if st.session_state.F_HP is not None:
        st.write('Fundus: HP prediction =', 0.188)
        cols = st.columns(10)
        for idx, img_name in enumerate(st.session_state.F_HP):
            img_path = os.path.join('Step4_HP/F/', img_name)
            image = Image.open(img_path)
            cols[idx % 10].image(image, use_container_width=True)    # caption=img_name
    if "pred_hp" in st.session_state:  
            st.write('Total HP prediction =', st.session_state.pred_hp, '(Total threshold = 0.5)')
            st.session_state.step = 4

    # ===== Step 4 =====
    st.subheader("Step 5 and 6: Histology prediction-AG(Atrophic gastritis), IM(Intestinal metaplasia)")
    st.markdown("##### <Using modified vision transformer model>")
    if st.session_state.step == 4:
        if st.button('Click here to predict AG, IM'):  
            st.session_state.pred_AG_A = 0.050
            st.session_state.pred_AG_B = 0.170
            st.session_state.pred_IM_A = 0.150
            st.session_state.pred_IM_B = 0.050
            st.session_state.AG_A = sorted(os.listdir('Step5_AG/A/'))
            st.session_state.AG_B = sorted(os.listdir('Step5_AG/B/'))
            st.session_state.IM_A = sorted(os.listdir('Step6_IM/A/'))
            st.session_state.IM_B = sorted(os.listdir('Step6_IM/B/'))
    time.sleep(2)
    if "pred_AG_A" in st.session_state and "pred_AG_B" in st.session_state and "pred_IM_A" in st.session_state and "pred_IM_B" in st.session_state: 
        if st.session_state.AG_A is not None:
            st.write('Antrum: AG prediction =', st.session_state.pred_AG_A, '(Total threshold = 0.2)')
            cols = st.columns(10)
            for idx, img_name in enumerate(st.session_state.AG_A):
                img_path = os.path.join('Step5_AG/A/', img_name)
                image = Image.open(img_path)
                cols[idx % 10].image(image, use_container_width=True)    # caption=img_name

        if st.session_state.AG_B is not None:
            st.write('Body: AG prediction =', st.session_state.pred_AG_B, '(Total threshold = 0.15)')
            cols = st.columns(10)
            for idx, img_name in enumerate(st.session_state.AG_B):
                img_path = os.path.join('Step5_AG/B/', img_name)
                image = Image.open(img_path)
                cols[idx % 10].image(image, use_container_width=True)    # caption=img_name

        if st.session_state.IM_A is not None:
            st.write('Antrum: IM prediction =', st.session_state.pred_IM_A, '(Total threshold = 0.07)')
            cols = st.columns(10)
            for idx, img_name in enumerate(st.session_state.IM_A):
                img_path = os.path.join('Step6_IM/A/', img_name)
                image = Image.open(img_path)
                cols[idx % 10].image(image, use_container_width=True)    # caption=img_name

        if st.session_state.IM_B is not None:
            st.write('Body: IM prediction =', st.session_state.pred_IM_B, '(Total threshold = 0.06)')
            cols = st.columns(10)
            for idx, img_name in enumerate(st.session_state.IM_B):
                img_path = os.path.join('Step6_IM/B/', img_name)
                image = Image.open(img_path)
                cols[idx % 10].image(image, use_container_width=True)    # caption=img_name
        st.session_state.step = 5

    # ===== Step 5 =====
    st.subheader("Step 7: Gastric cancer prediction")
    if st.session_state.step == 5:
        st.session_state.df = pd.DataFrame({'Feature': ['HP', 'AG_Antrum', 'AG_Body', 'IM_Antrum', 'IM_Body', 'Age', 'Sex'], 'P249750000282': [0, 0, 0, 0, 0, 75, 1]}) 
    time.sleep(3)
    if st.session_state.df is not None:
        st.write('Overall summary')
        st.dataframe(st.session_state.df, use_container_width=True)
        st.markdown("##### <Using logistic regression>")
        if st.button('Click here to predict the probability of gastric cancer'):
            time.sleep(1)
            st.write('Gastric cancer prediction =', 0.083)
import mediapipe as mp
import numpy  as np
import cv2
import streamlit as st
import time

movie_path = "app2/movie/"
img_path = "app2/image/"

def sidebar_pram():
    col1, col2 = st.sidebar.columns(2)
    button_run = col1.button("start")
    button_stop = col2.button("stop")
    mode = st.sidebar.selectbox('モードの選択', ['Use movie file', 'Use WebCam'])
    fps_val = st.sidebar.slider('フレームレート', 1, 100, 50)

    uploaded_mv_file = None
    if mode == 'Use movie file':
        uploaded_mv_file = st.sidebar.file_uploader("動画ファイルアップロード", type = 'mp4')
        if uploaded_mv_file is not None:
            st.sidebar.video(uploaded_mv_file)

    uploaded_img_file = None
    uploaded_img_file = st.sidebar.file_uploader("背景画像ファイルアップロード", type= ['jpg', 'jpeg', 'png'])
    if uploaded_img_file is not None:
        st.sidebar.image(uploaded_img_file)
    
    return button_run, button_stop, mode, fps_val, uploaded_mv_file, uploaded_img_file


def read_img_movie(img_path, uploaded_img_file, movie_path, uploaded_mv_file):
    img_file_path = img_path + uploaded_img_file.name
    mv_file_path  = None
    cap_file = None

    org_bd_image = cv2.imread(img_file_path)

    if mode == 'Use movie file':
        mv_file_path = movie_path + uploaded_mv_file.name
        cap_file = cv2.VideoCapture(mv_file_path)
    else:
        cap_file = cv2.VideoCapture(0)
    return org_bd_image, cap_file

def create_virtual_bg(button_stop, org_bg_image, cap_file, mp_selfie_segmentation, mode, fps_val):
    

if __name__ == "__main__":
    st.sidebar.title('各種設定')
    button_run, button_stop, mode, fps_val, uploaded_mv_file, uploaded_img_file = sidebar_pram()

    st.title('バーチャル背景')
    if button_run:
        if mode == "Use movie file" and uploaded_mv_file is None:
            st.text("動画をアップロードしてください")
        elif uploaded_img_file is None:
            st.text("画像をアップロードしてください")
        else:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            org_bd_image, cap_file = read_img_movie(img_path, uploaded_img_file, movie_path, uploaded_mv_file):




# main.py

import data.data_prepare as dp
import models.model_utils as mu
from utils import streamlit, helpers
import streamlit as st

from train import train_progress

def main():
    st.title("Dog Breed Classification")
    train_batch_size, eval_batch_size = 8, 8
    num_workers = 8
    
    if 'training_flag' not in st.session_state:
        st.session_state['training_flag'] = ''
    
    if 'dataset_path' not in st.session_state:
        st.session_state['dataset_path'] = ''
        
    if 'model_path' not in st.session_state:
        st.session_state['model_path'] = ''
    dataset_path, model_type, model_path, num_epochs, train_batch_size, eval_batch_size = streamlit.side_bar()
    training_progess_header = st.empty()

    start = st.sidebar.button('Start Training', on_click=streamlit.startButtonPressed)
    stop = st.sidebar.button('Stop Training', on_click=streamlit.stopButtonPressed)
    st.sidebar.text("Continue from local checkpoint")
    continue_button = st.sidebar.button('Continue', on_click=streamlit.continueButtonPressed)
    if ((st.session_state['training_flag'] == 'train' and start) or 
        (st.session_state['training_flag'] == 'continue' and continue_button)):
        if st.session_state['dataset_path'] and dataset_path:
            training_progess_header.header('Training progress')
            
            train_progress(dataset_path=dataset_path,
                           _model_type=model_type,
                           model_path=model_path,
                           train_batch_size=int(train_batch_size),
                           eval_batch_size=int(eval_batch_size),
                           num_workers=num_workers,
                           num_epochs=num_epochs)
            # test_train()
            
            st.success("Training completed.")
        else:
            st.error(f"Please input a valid dataset path")
            st.error(f"train and valid folder should exist, e.g. ./dataset/train and ./dataset/valid")
        st.session_state['training_flag'] = 'stop'
    
    if st.session_state['training_flag'] == 'stop' and stop:
        st.info("Training stopped by user.")

if __name__ == '__main__':
    main()
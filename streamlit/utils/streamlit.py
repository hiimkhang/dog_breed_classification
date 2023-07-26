# utils/streamlit.py

import streamlit as st
import time

import pandas as pd
import utils.helpers as helpers
import data.data_prepare as dp
import models.model_utils as mu
import math

class StreamlitProgressBar:
    def __init__(self, current_iteration, total_iterations, desc):
        self.current_iteration = current_iteration
        self.total_iterations = total_iterations
        self.desc = desc
        self.desc_text = st.text(desc)
        self.progress_bar = st.progress(current_iteration / total_iterations)
        self.start_time = time.time()
        self.detail_text = st.empty()
    
    def batchUpdate(self, batch, epoch):
        completed_percent = batch / self.total_iterations
        remaining_percent = 1 - completed_percent
        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / completed_percent) * remaining_percent
        self.progress_bar.progress(completed_percent)
        self.detail_text.text(f"Epoch {epoch+1} {self.desc} | "
                             f"Batch: {batch}/{self.total_iterations} | "
                             f"Elapsed time: {elapsed_time:.2f}s | "
                             f"Estimated time remaining: {remaining_time:.2f}s")
    
    def epochUpdate(self, epoch):
        completed_percent = epoch / self.total_iterations
        remaining_percent = 1 - completed_percent
        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / completed_percent) * remaining_percent
        self.progress_bar.progress(completed_percent)
        self.detail_text.text(f"Epoch: {epoch}/{self.total_iterations} | "
                             f"Elapsed time: {elapsed_time:.2f}s | "
                             f"Estimated time remaining: {remaining_time:.2f}s")
    
    def updateDescription(self, text):
        self.desc_text.text(text)
    
    def close(self):
        self.detail_text.empty()
        self.desc_text.empty()
        self.progress_bar.empty()
        
def side_bar():
    st.sidebar.header("Configuration")
    model_path = ""
    
    # Input dataset path
    dataset_path = st.sidebar.text_input("Dataset path", on_change=resetTrainingFlag)
    if dataset_path:
        if helpers.check_dataset_path(dataset_path):
            st.sidebar.success("Dataset path exists")
            st.session_state['dataset_path'] = dataset_path
        else:
            st.sidebar.error("Invalid dataset path")
    
    # Select model type
    model_type = st.sidebar.selectbox("Model type", 
                                      options=mu.get_available_modeltypes(), 
                                      on_change=resetTrainingFlag,
                                      index=0)
    
    # If a model is selected, enable input for model path
    if model_type:
        model_path = st.sidebar.text_input("Model path", on_change=resetTrainingFlag)
        if model_path:
            if helpers.check_pretrained_model_path(model_path):
                st.sidebar.success("Model path exists")
                st.info(f"Using pretrained model of {model_type}")
                st.session_state['model_path'] = model_path
            else:
                st.sidebar.error("Invalid model path")
    
    st.sidebar.text(f"Device using: {str(helpers.get_device())}")
    
    num_epochs = st.sidebar.slider("Number of epochs", min_value=1, max_value=30, value=5, step=1)
    train_batch_size = st.sidebar.selectbox("Train batch size", options=[8, 16, 32, 64, 128, 256, 512])
    eval_batch_size = st.sidebar.selectbox("Test batch size", options=[8, 16, 32, 64, 128, 256, 512])
    
    return dataset_path, str(model_type), model_path, num_epochs, str(train_batch_size), str(eval_batch_size)

def next_power_of_2(x):
    return 2 ** math.ceil(math.log2(x))
 
def resetTrainingFlag():
    st.session_state['training_flag'] = ''
    
def startButtonPressed():
    st.session_state['training_flag'] = 'train'
    helpers.log_message("Start button pressed")

def stopButtonPressed():
    st.session_state['training_flag'] = 'stop'
    helpers.log_message("Stop button pressed")
    
def continueButtonPressed():
    st.session_state['training_flag'] = 'continue'
    helpers.log_message("Continue button pressed")
    
def plot_loss_accuracy_curve(results: pd.DataFrame):
    # Plot the loss curve   
    st.subheader("Loss Curve")
    st.line_chart(results[['Epoch', 'Train loss', 'Test loss']], x='Epoch', y=['Train loss', 'Test loss'])

    # Plot the accuracy curve
    st.subheader("Accuracy Curve")
    st.line_chart(results[['Epoch', 'Train accuracy', 'Test accuracy']], x='Epoch', y=['Train accuracy', 'Test accuracy'])

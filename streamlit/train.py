import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import streamlit, helpers
import models.model_utils as mu
import data.data_prepare as dp
import os
import pandas as pd

saved_models_dir = './saved_models'
results_info = []


def train_step(model: nn.Module,
               train_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int) -> tuple[float, float]:
    model.train()

    train_loss, train_accuracy = 0., 0.
    progress_bar = streamlit.StreamlitProgressBar(
        0, len(train_loader), desc="Train")

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)  # forward pass

        loss = loss_fn(y_pred, y)  # compute loss
        train_loss += loss.item()

        optimizer.zero_grad()  # reset gradients

        loss.backward()  # backward pass

        optimizer.step()  # update parameters

        y_pred_labels = torch.argmax(torch.softmax(
            y_pred, dim=1), dim=1)  # compute accuracy
        train_accuracy += (y_pred_labels == y).sum().item() / len(y_pred)
        progress_bar.batchUpdate(batch+1, epoch)

    progress_bar.close()
    return train_loss/len(train_loader), train_accuracy/len(train_loader)


def eval_step(model: nn.Module,
              test_loader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device,
              epoch: int) -> tuple[float, float]:

    model.eval()

    test_loss, test_accuracy = 0., 0.
    progress_bar = streamlit.StreamlitProgressBar(
        0, len(test_loader), desc="Test")

    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)  # forward pass

            loss = loss_fn(y_pred_logits, y)  # compute loss
            test_loss += loss.item()

            y_pred_labels = y_pred_logits.argmax(dim=1)  # compute accuracy
            test_accuracy += (y_pred_labels == y).sum().item() / \
                len(y_pred_labels)
            progress_bar.batchUpdate(batch+1, epoch)

    progress_bar.close()
    return test_loss/len(test_loader), test_accuracy/len(test_loader)


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer,
                start_epoch: int,
                epochs: int,
                device: torch.device,
                results):

    global results_info
    model.to(device)

    progress_bar = streamlit.StreamlitProgressBar(
        start_epoch, epochs, desc=f"The model is training epoch {start_epoch+1}/{epochs}")

    for epoch in range(start_epoch, epochs):

        progress_bar.updateDescription(
            f"The model is training epoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device, epoch)
        test_loss, test_acc = eval_step(
            model, test_dataloader, loss_fn, device, epoch)

        info = streamlit.st.info(f"Epoch {epoch+1}/{epochs} | "
                                 f"Train loss: {train_loss:.4f} | "
                                 f"Train accuracy: {train_acc:.4f} | "
                                 f"Test loss: {test_loss:.4f} | "
                                 f"Test accuracy: {test_acc:.4f}\n")

        results_info.append(info)

        results['Train loss'].append(train_loss)
        results['Train accuracy'].append(train_acc)
        results['Test loss'].append(test_loss)
        results['Test accuracy'].append(test_acc)

        mu.save_checkpoint(model=model,
                           model_type=model_type,
                           optimizer=optimizer,
                           epoch=epoch,
                           num_epochs=epochs,
                           result=results)

        progress_bar.epochUpdate(epoch+1)

    progress_bar.close()

    return results


def train_progress(dataset_path: str,
                   _model_type: str,
                   model_path: str,
                   train_batch_size: int,
                   eval_batch_size: int,
                   num_workers: int,
                   num_epochs=5):

    global model_type, results_info

    results = {'Train loss': [],
               'Train accuracy': [],
               'Test loss': [],
               'Test accuracy': []
               }

    model_type = _model_type
    dataloader, image_datasets = dp.get_data_loaders(
        dataset_path, train_batch_size, eval_batch_size, num_workers)
    in_channels = dp.get_input_tensor_shape(dataloader)[0]
    num_classes = dp.get_output_size(image_datasets)

    model = mu.load_model(path=model_path,
                          model_type=model_type,
                          in_channels=in_channels,
                          num_classes=num_classes,
                          device=helpers.get_device())

    optimizer = torch.optim.SGD(model.parameters(),
                                1e-2,
                                momentum=0.9,
                                weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0

    if streamlit.st.session_state['training_flag'] == 'continue':
        print("\nContinue")
        start_epoch, result = mu.load_checkpoint(model_type=model_type,
                                                 model=model,
                                                 optimizer=optimizer)
        if (start_epoch >= num_epochs):
            streamlit.st.info("Last checkpoint reaches the target epoch!")
            start_epoch = 0
            result.clear()
        results.update(result)
        streamlit.st.info(f"Resuming from epoch {start_epoch + 1}")

    train_model(model=model,
                train_dataloader=dataloader['train'],
                test_dataloader=dataloader['valid'],
                loss_fn=loss_fn,
                optimizer=optimizer,
                start_epoch=start_epoch,
                epochs=num_epochs,
                device=helpers.get_device(),
                results=results)
    # train_result = test_train()
    # results.update(train_result)
    dataset_name = dataset_path.split("/")[-1]
    save_name = f"{model_type}_{dataset_name}"
    # Save the trained model
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir, exist_ok=True)
    torch.save(model.state_dict(),
               f"{os.path.join(saved_models_dir, save_name)}.pth")
    helpers.log_message(
        f"Model saved to {os.path.join(saved_models_dir, save_name)}.pth")

    streamlit.st.info(
        f"Model saved to {os.path.join(saved_models_dir, save_name)}.pth")

    pd_result = pd.DataFrame(results, index=range(1, num_epochs+1))
    pd_result.reset_index(inplace=True)
    pd_result.rename(columns={'index': 'Epoch'}, inplace=True)

    streamlit.st.table(pd_result)
    for info in results_info:
        info.empty()
    results_info.clear()

    streamlit.plot_loss_accuracy_curve(pd_result)

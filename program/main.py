# main.py

import argparse
import torch
import data_prepare as dp
import model_utils as mu
from train import train_model
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="No.")
    parser.add_argument("-d", "--dataset", type=str, default='../small_dataset', help="Path to the dataset.")
    parser.add_argument("-a", "--arch", type=str, default="resnet50", help="Name of the model architecture.")
    parser.add_argument("-r", "--resume", action="store_true", default=False, help="Resume from local checkpoint.")
    parser.add_argument("-p", "--pretrained", type=str, required=False, help="Path to pre-trained model (if any).")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--train_batch", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--test_batch", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    checkpoint_dir = "./checkpoints"
    saved_models_dir = "./saved_models"
    saved_plots_dir = "./plots"
    
    if args.arch == "": args.arch = "resnet50"
    if args.dataset == "": args.dataset = "../small_dataset"
    dataset_name = args.dataset.split("/")[-1]
    
    results = {'Train loss': [],
            'Train accuracy': [], 
            'Test loss': [], 
            'Test accuracy': []
        } 

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Load the dataset and create data loaders
    dataloader, image_datasets = dp.get_data_loaders(args.dataset, args.train_batch, args.test_batch, args.workers)
    in_channels = dp.get_input_tensor_shape(dataloader)[0]
    out_features = dp.get_output_size(image_datasets)

    # Load the model
    model = mu.load_model(model_type=args.arch, 
                          in_channels=in_channels, 
                          num_classes=out_features, 
                          device=mu.get_device(), 
                          path=args.pretrained)
        
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=1e-4)
    start_epoch = 0
    
    if args.resume and args.arch in mu.get_available_modeltypes():
        start_epoch, results = mu.load_checkpoint(model_type=args.arch,
                                                model=model, 
                                                optimizer=optimizer,
                                                dataset_name=dataset_name)
    print(args.arch, args.dataset)
    # Training
    train_results = train_model(model=model,
                                model_type=args.arch,
                          train_dataloader=dataloader['train'],
                          test_dataloader=dataloader['valid'],
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          start_epoch=start_epoch,
                          epochs=args.epochs,
                          device=mu.get_device(),
                          results=results,
                          dataset_name=dataset_name)
    
    results.update(train_results)
    mu.log_message(f"Model: {args.arch}\nResults: {results}", is_print=False)
    mu.log_message(f"Trainning complete!", is_print=True)
    
    save_name = f"{args.arch}_{dataset_name}"
    # Save the trained model
    mu.check_and_create_dir(saved_models_dir)
    torch.save(model.state_dict(), f"{os.path.join(saved_models_dir, save_name)}.pth")
    mu.log_message(f"Model saved to {os.path.join(saved_models_dir, save_name)}.pth", is_print=True)
    
    # Save plots in './plots' directory
    mu.plot_loss_accuracy_curve(results, saved_plots_dir, save_name)
    mu.log_message(f"Plots saved to {saved_plots_dir}!", is_print=True)

if __name__ == "__main__":
    main()

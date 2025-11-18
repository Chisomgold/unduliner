#!/usr/bin/env python

import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import random
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch import optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, matthews_corrcoef, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import warnings
import itertools
import re
from torch import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

import logging
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Check for multiple GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
logger.info(f'torch device: {device}, Available GPUs: {n_gpus}')

import psutil
import time

start = time.time()

# Implementing seeding for reproducibility/debugging
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#  Network architecture
class Net(nn.Module):
    def __init__(self, input_size, num_layers):
        super(Net, self).__init__()
        layers = []
        # initial input and output channels
        c_in = 3
        c_out = 256

        for i in range(num_layers):
            layers.append(self.conv_block(c_in=c_in, c_out=c_out, dropout=0.4 if i == 0 else 0.25, kernel_size=3 if i != 0 else 5, stride=1, padding=1 if i != 0 else 2))
            c_in = c_out
            c_out = c_out // 2  # Halve the number of output channels after each layer

            # Add max pooling after every two layers
            if (i + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(in_channels=c_in, out_channels=2, kernel_size=input_size // (2 ** ((num_layers + 1) // 2)), stride=1, padding=0))
        layers.append(nn.Flatten())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

        return seq_block


def accuracy(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_tag, dim=1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs, device, hyperparams, use_amp=True):

    if n_gpus > 1:
        model = DataParallel(model)
        logger.info(f"Using DataParallel with {n_gpus} GPUs")
    model.to(device)

    # Mixed precision training
    scaler = GradScaler() if use_amp else None

    # Early stopping parameters
    best_val_loss = float('inf')
    best_f1_score = 0.0
    patience_counter = 0
    patience_limit = 15

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model = {}
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        training_acc = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast(device_type=str(device)):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            training_loss += loss.item()
            training_acc += accuracy(outputs, targets).item()

            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        #Validation
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        valid_loss = 0.0
        valid_acc = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                if use_amp:
                    with autocast(device_type=str(device)):
                        outputs = model(inputs)
                        val_loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)

                valid_loss += val_loss.item()
                valid_acc += accuracy(outputs, targets).item()

                probs = F.softmax(outputs, 1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate epoch metrics
        avg_train_loss = training_loss / len(train_loader)
        avg_val_loss = valid_loss / len(val_loader)
        avg_train_acc = training_acc / len(train_loader)
        avg_val_acc = valid_acc / len(val_loader)


        # Calculate F1 score
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        f1_score = report['macro avg']['f1-score']
        mcc = matthews_corrcoef(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)

        logger.info(f'Epoch {epoch+1:02}: | Train loss: {avg_train_loss:.3f} | Val loss: {avg_val_loss:.3f} | '
              f'Train acc: {avg_train_acc:.3f} | Val acc: {avg_val_acc:.3f} | F1: {f1_score:.3f}')

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        # checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if f1_score > best_f1_score:
            best_f1_score = f1_score

            # Save best model
            model_name = f"model_{hyperparams}_f1_{int(f1_score*100)}.pth"
            if n_gpus > 1:
                best_model_state = model.module.state_dict().copy()
            else:
                best_model_state = model.state_dict().copy()

            best_model = {'classification_report': report, 'all_labels': all_labels.copy(),
                          'all_preds': all_preds.copy(), 'mcc' : mcc, 'precision': precision,
                          'recall': recall, 'pr_auc': pr_auc}

            torch.save(best_model_state, model_name)
            logger.info(f"Saved best model with F1 score: {f1_score:.4f}")

        # Early stopping
        if patience_counter >= patience_limit:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break

        # Additional stopping criteria
        if avg_train_acc > 99.0 and avg_val_acc > 99.0:
            logger.info('Stopping: Accuracy > 99%')
            break

    # Final report
    logger.info(classification_report(best_model['all_labels'], best_model['all_preds'], target_names=['neg', 'pos'], labels=[0, 1]))

    logger.info(f"\n === Metrics for best model - F1: {best_f1_score:.4f}, MCC: {best_model['mcc']:.4f}, PR_AUC: {best_model['pr_auc']:.4f} === \n")

    return model_name, best_f1_score, history

def test(model, testloader, device, use_amp=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if use_amp:
                with autocast(device_type=str(device)):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    report = classification_report(all_labels, all_preds, output_dict=True)
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    f1_score = report['macro avg']['f1-score']

    return f1_score, report, mcc, pr_auc, precision, recall, all_labels, all_preds

def create_data_loaders(batch_size, num_workers=8, seed=None):
    if seed is not None:
        set_seed(seed)

    # Data directories
    train_dir = '/lustre/user/IgvCmapimages/newtrain'
    val_dir = '/lustre/user/IgvCmapimages/newval'
    test_dir = '/lustre/user/IgvCmapimages/newtest'

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])

    trainset = ImageFolder(root=train_dir, transform=transform)
    valset = ImageFolder(root=val_dir, transform=transform)
    testset = ImageFolder(root=test_dir, transform=transform)

    #class weights for imbalanced dataset
    targets = np.array(trainset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    weights = [class_weights[label] for label in targets]

    print(f'\nDataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}')
    print(f"Class distribution - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")

    train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(trainset), replacement=True)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader, class_weights

def main():
    #Hyperparameters
    batch_sizes = [32, 64, 128]
    learning_rates = [0.0003, 0.0001, 0.005, 0.001, 0.003]
    weight_decays = [1e-10]
    num_layers_options = [4, 5]
    epochs = 200

    # Determine optimal number of workers
    num_workers = min(16, os.cpu_count())
    logger.info(f"Using {num_workers} workers for data loading")

    all_models = []
    total_models_trained = 0

    for num_layers in num_layers_options:
        for batch_size, lr, weight_decay in itertools.product(batch_sizes, learning_rates, weight_decays):
            logger.info(f"\n{'='*50}")
            logger.info(f"Training configuration {total_models_trained + 1}")
            logger.info(f"{'='*50}")

            # Create data loaders
            train_loader, val_loader, test_loader, class_weights = create_data_loaders(
                batch_size, num_workers=num_workers, seed=42
            )

            # Initialize model
            model = Net(input_size=224, num_layers=num_layers)
            logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Training
            hyperparams = f"Adam_lr{lr}_batch{batch_size}_wd{weight_decay}_layers{num_layers}"
            logger.info(f'Training with {hyperparams}')

            model_name, val_f1_score, history = train_and_evaluate(
                model, train_loader, val_loader, criterion, optimizer, epochs, device, hyperparams
            )

            all_models.append((model_name, val_f1_score, hyperparams, num_layers))
            total_models_trained += 1

            # Save training history
            pd.DataFrame(history).to_csv(f'training_history_{hyperparams}.csv', index=False)

    # Testing phase
    logger.info(f"\n{'='*50}")
    logger.info("TESTING PHASE")
    logger.info(f"{'='*50}")

    test_results = []

    for model_name, val_f1_score, hyperparams, num_layers in all_models:
        model = Net(input_size=224, num_layers=num_layers)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        start_infer = time.time()

        logger.info(f'\n=== Testing model: {model_name} ===')
        test_f1_score, report, test_mcc, test_pr_auc, test_precision, test_recall, test_label, test_pred = test(
            model, test_loader, device
        )

        end_infer = time.time()
        infer_time_per_sample = (end_infer - start_infer) / len(test_label)
        logger.info(f"Inference time per sample - {model_name}: {infer_time_per_sample*1000:.2f} ms")

        logger.info(f'Test F1: {test_f1_score:.4f}, MCC: {test_mcc:.4f}, PR-AUC: {test_pr_auc:.4f}')

        test_results.append((model_name, test_f1_score, test_mcc, test_pr_auc, test_precision, test_recall, test_label, test_pred))


    # Save results
    test_results_sorted = sorted(test_results, key=lambda x: x[1], reverse=True)[:10]

    df = pd.DataFrame([t[:4] for t in test_results_sorted], columns=['Model name', 'F1 score', 'MCC', 'PR_AUC'])
    logger.info("\n=== Top Models Metrics ===")
    logger.info(df[['Model name', 'F1 score', 'MCC', 'PR_AUC']])

    output_file = 'top_models_metrics.tsv'
    df.to_csv(output_file, sep='\t', index=False)
    logger.info(f"\nResults saved to: {output_file}")

    # CPU memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_gb = mem_info.rss / (1024 ** 3)
    logger.info(f"Peak CPU RAM usage: {ram_gb:.2f} GB")

    # GPU memory usage (if available)
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"Peak GPU memory used: {gpu_mem_gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats()  # optional, resets tracker for next run

    # Plot PR curves for top 5 models
    plt.figure(figsize=(10, 8))
    for i, (model_name, _, _, _, precision, recall, _, _) in enumerate(test_results_sorted[:5]):
        plt.plot(recall, precision, label=f'Model {i+1} (F1: {test_results_sorted[i][1]:.3f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Top 5 Models")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig("top_models_pr_curve.png", dpi=300, bbox_inches='tight')
    logger.info("PR curve plot saved")

    #confusion matrix
    for i, (model_name, _, _, _, _, _, y_true, y_pred) in enumerate(test_results_sorted[:5]):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Model {i+1}: {model_name}")
        plt.savefig(f"confusion_matrix_model_{i+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved for Model {i+1}: {model_name}")

    # Training time
    end = time.time()
    elapsed_seconds = end - start
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60

    logger.info(f"\n{'='*50}")
    logger.info(f"Total training time: {elapsed_hours:.2f} hours ({elapsed_minutes:.1f} minutes)")
    logger.info(f"Total models trained: {total_models_trained}")
    logger.info(f"Average time per model: {elapsed_minutes/total_models_trained:.1f} minutes")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()

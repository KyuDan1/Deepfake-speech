"""
RawNet2 Training Script for ASVspoof5 Dataset

Based on the original ASVspoof2021 baseline by Hemlata Tak.
Modified for ASVspoof5 dataset format.

Usage:
    python train_asvspoof5.py --database_path /path/to/ASVspoof5/
    python train_asvspoof5.py --eval --model_path models/best_model.pth
"""
import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'asvspoof2021_baseline' / '2021' / 'LA' / 'Baseline-RawNet2'))

from data_utils_asvspoof5 import (
    parse_asvspoof5_protocol,
    Dataset_ASVspoof5_train,
    Dataset_ASVspoof5_eval,
    get_class_weights
)
from model import RawNet

# Try importing tensorboard
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        print("Warning: TensorBoard not available. Training will continue without logging.")


def set_random_seed(seed, args=None):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if args and hasattr(args, 'cudnn_deterministic_toggle') and args.cudnn_deterministic_toggle:
            torch.backends.cudnn.deterministic = True
        if args and hasattr(args, 'cudnn_benchmark_toggle') and args.cudnn_benchmark_toggle:
            torch.backends.cudnn.benchmark = True


def evaluate_accuracy(dev_loader, model, device):
    """
    Evaluate model accuracy on validation set.

    Args:
        dev_loader: DataLoader for validation data
        model: RawNet model
        device: torch device

    Returns:
        Accuracy percentage
    """
    num_correct = 0.0
    num_total = 0.0
    model.eval()

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

    return 100 * (num_correct / num_total)


def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER).

    Args:
        scores: Model prediction scores
        labels: Ground truth labels (1=bonafide, 0=spoof)

    Returns:
        EER value
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where FPR == FNR
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2

    return eer * 100  # Return as percentage


def evaluate_eer(dev_loader, model, device):
    """
    Evaluate model EER on validation set.

    Args:
        dev_loader: DataLoader for validation data
        model: RawNet model
        device: torch device

    Returns:
        EER percentage
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)

            # Use score for bonafide class (index 1)
            batch_score = batch_out[:, 1].data.cpu().numpy()
            all_scores.extend(batch_score.tolist())
            all_labels.extend(batch_y.numpy().tolist())

    return compute_eer(np.array(all_scores), np.array(all_labels))


def produce_evaluation_file(dataset, model, device, save_path):
    """
    Generate evaluation scores file.

    Args:
        dataset: Evaluation dataset
        model: RawNet model
        device: torch device
        save_path: Path to save scores
    """
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()

    with torch.no_grad():
        for batch_x, utt_id in data_loader:
            fname_list = []
            score_list = []
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)

            # Score is the log probability of bonafide class
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list, score_list):
                    fh.write(f'{f} {cm}\n')

    print(f'Scores saved to {save_path}')


def train_epoch(train_loader, model, optimizer, criterion, device):
    """
    Train for one epoch.

    Args:
        train_loader: DataLoader for training data
        model: RawNet model
        optimizer: Optimizer
        criterion: Loss function
        device: torch device

    Returns:
        Tuple of (average_loss, train_accuracy)
    """
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.train()

    for ii, (batch_x, batch_y) in enumerate(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # Forward pass
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)

        # Compute accuracy
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

        # Progress display
        if (ii + 1) % 10 == 0:
            sys.stdout.write(f'\r  Batch {ii+1}/{len(train_loader)} - Acc: {(num_correct/num_total)*100:.2f}%')

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100

    return running_loss, train_accuracy


def main():
    parser = argparse.ArgumentParser(description='RawNet2 Training for ASVspoof5')

    # Dataset paths
    parser.add_argument('--database_path', type=str,
                        default='/mnt/tmp/Deepfake-speech/data/ASVspoof5/',
                        help='Path to ASVspoof5 dataset root directory')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for optimizer (default: 0.0001)')

    # Model settings
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Run evaluation only')
    parser.add_argument('--eval_output', type=str, default='eval_scores.txt',
                        help='Path to save evaluation scores')

    # Loss function
    parser.add_argument('--loss', type=str, default='weighted_CCE',
                        choices=['weighted_CCE', 'CCE'],
                        help='Loss function (default: weighted_CCE)')
    parser.add_argument('--auto_weight', action='store_true', default=False,
                        help='Automatically compute class weights from data')

    # Backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false',
                        default=True,
                        help='Use cudnn-deterministic (default: True)')
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true',
                        default=False,
                        help='Use cudnn-benchmark (default: False)')

    # Data loading
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed, args)

    # Model configuration (same as original RawNet2)
    model_config = {
        'nb_samp': 64600,
        'first_conv': 1024,
        'in_channels': 1,
        'filts': [20, [20, 20], [20, 128], [128, 128]],
        'blocks': [2, 4],
        'nb_fc_node': 1024,
        'gru_node': 1024,
        'nb_gru_layer': 3,
        'nb_classes': 2
    }

    # Paths
    database_path = Path(args.database_path)
    train_audio_dir = database_path / 'flac_T'
    dev_audio_dir = database_path / 'flac_D'
    eval_audio_dir = database_path / 'flac_E_eval'

    train_protocol = database_path / 'ASVspoof5.train.tsv'
    dev_protocol = database_path / 'ASVspoof5.dev.track_1.tsv'
    eval_protocol = database_path / 'ASVspoof5.eval.track_1.tsv'

    # Create model save directory
    model_tag = f'rawnet2_asvspoof5_{args.loss}_{args.num_epochs}_{args.batch_size}_{args.lr}'
    if args.comment:
        model_tag = f'{model_tag}_{args.comment}'

    model_save_path = Path(__file__).parent / 'models' / model_tag
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Initialize model
    model = RawNet(model_config, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f'Model parameters: {nb_params:,}')
    model = model.to(device)

    # Load pretrained model if specified
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Model loaded from: {args.model_path}')

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Evaluation mode
    if args.eval:
        print('Running evaluation...')

        if eval_protocol.exists():
            file_eval = parse_asvspoof5_protocol(eval_protocol, is_eval=True)
            print(f'Number of eval trials: {len(file_eval)}')
            eval_set = Dataset_ASVspoof5_eval(
                list_IDs=file_eval,
                base_dir=str(eval_audio_dir)
            )
            produce_evaluation_file(eval_set, model, device, args.eval_output)
        else:
            print(f'Evaluation protocol not found: {eval_protocol}')

        sys.exit(0)

    # Load training data
    print(f'Loading training data from: {train_protocol}')
    d_label_trn, file_train = parse_asvspoof5_protocol(train_protocol, is_eval=False)
    print(f'Number of training trials: {len(file_train)}')

    train_set = Dataset_ASVspoof5_train(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=str(train_audio_dir)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Load validation data
    print(f'Loading validation data from: {dev_protocol}')
    d_label_dev, file_dev = parse_asvspoof5_protocol(dev_protocol, is_eval=False)
    print(f'Number of validation trials: {len(file_dev)}')

    dev_set = Dataset_ASVspoof5_train(
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=str(dev_audio_dir)
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Loss function
    if args.loss == 'weighted_CCE':
        if args.auto_weight:
            weight = get_class_weights(d_label_trn).to(device)
        else:
            # Default weights (favoring bonafide detection)
            weight = torch.FloatTensor([0.1, 0.9]).to(device)
            print(f'Using default class weights: [Spoof=0.1, Bonafide=0.9]')
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
        print('Using unweighted CrossEntropyLoss')

    # TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = Path(__file__).parent / 'logs' / model_tag
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f'TensorBoard logs: {log_dir}')

    # Training loop
    print(f'\nStarting training for {args.num_epochs} epochs...')
    print('=' * 60)

    best_acc = 0.0
    best_eer = 100.0

    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch + 1}/{args.num_epochs}')

        # Train
        running_loss, train_accuracy = train_epoch(
            train_loader, model, optimizer, criterion, device
        )

        # Validate
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)

        # Compute EER (optional, can be slow)
        try:
            valid_eer = evaluate_eer(dev_loader, model, device)
            eer_str = f'{valid_eer:.2f}%'
        except Exception as e:
            valid_eer = None
            eer_str = 'N/A'

        # Log to TensorBoard
        if writer:
            writer.add_scalar('train/accuracy', train_accuracy, epoch)
            writer.add_scalar('train/loss', running_loss, epoch)
            writer.add_scalar('valid/accuracy', valid_accuracy, epoch)
            if valid_eer is not None:
                writer.add_scalar('valid/eer', valid_eer, epoch)

        # Print progress
        print(f'\n  Loss: {running_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
              f'Valid Acc: {valid_accuracy:.2f}% | Valid EER: {eer_str}')

        # Save best model
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            best_model_path = model_save_path / 'best_acc_model.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'  -> Best accuracy model saved: {best_model_path}')

        if valid_eer is not None and valid_eer < best_eer:
            best_eer = valid_eer
            best_eer_model_path = model_save_path / 'best_eer_model.pth'
            torch.save(model.state_dict(), best_eer_model_path)
            print(f'  -> Best EER model saved: {best_eer_model_path}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = model_save_path / f'epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  -> Checkpoint saved: {checkpoint_path}')

    # Save final model
    final_model_path = model_save_path / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'\nTraining complete!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Best validation EER: {best_eer:.2f}%')
    print(f'Final model saved: {final_model_path}')

    if writer:
        writer.close()


if __name__ == '__main__':
    main()

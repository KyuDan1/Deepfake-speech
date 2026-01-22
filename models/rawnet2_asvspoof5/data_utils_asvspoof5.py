"""
ASVspoof5 Dataset Utilities for RawNet2 Training
Based on the original ASVspoof2019 data_utils.py by Hemlata Tak
"""
import os
import numpy as np
import torch
from torch import Tensor
import librosa
from torch.utils.data import Dataset


def parse_asvspoof5_protocol(protocol_path, is_eval=False):
    """
    Parse ASVspoof5 protocol file (TSV format).

    ASVspoof5 TSV format:
    speaker_id file_id gender - - - codec attack_type label -
    Example: T_4850 T_0000000000 F - - - AC3 A05 spoof -

    Args:
        protocol_path: Path to the protocol TSV file
        is_eval: If True, returns only file list (for evaluation without labels)

    Returns:
        If is_eval=False: (d_meta, file_list) where d_meta is {file_id: label} dict
        If is_eval=True: file_list only
    """
    d_meta = {}
    file_list = []

    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            speaker_id = parts[0]
            file_id = parts[1]
            gender = parts[2]
            # parts[3:6] are placeholders (-)
            codec = parts[6] if len(parts) > 6 else '-'
            attack_type = parts[7] if len(parts) > 7 else '-'
            label = parts[8] if len(parts) > 8 else 'bonafide'

            file_list.append(file_id)

            if not is_eval:
                # Label: 1 for bonafide, 0 for spoof
                d_meta[file_id] = 1 if label == 'bonafide' else 0

    if is_eval:
        return file_list
    else:
        return d_meta, file_list


def pad(x, max_len=64600):
    """
    Pad audio to fixed length using tile-and-truncate method.

    Args:
        x: Input audio array
        max_len: Target length (default: 64600 samples = ~4 sec at 16kHz)

    Returns:
        Padded audio array of length max_len
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Need to pad - tile the audio and truncate
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof5_train(Dataset):
    """
    ASVspoof5 Training/Development Dataset.

    Loads FLAC audio files and corresponding labels for training.
    """
    def __init__(self, list_IDs, labels, base_dir, cut=64600):
        """
        Args:
            list_IDs: List of file IDs (without extension)
            labels: Dictionary mapping file_id -> label (1=bonafide, 0=spoof)
            base_dir: Base directory containing audio files
            cut: Audio length to use (default: 64600 samples = ~4 sec at 16kHz)
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]

        # Load audio file
        audio_path = os.path.join(self.base_dir, f'{key}.flac')
        X, fs = librosa.load(audio_path, sr=16000)

        # Pad to fixed length
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)

        # Get label
        y = self.labels[key]

        return x_inp, y


class Dataset_ASVspoof5_eval(Dataset):
    """
    ASVspoof5 Evaluation Dataset.

    Loads FLAC audio files for evaluation (returns file IDs instead of labels).
    """
    def __init__(self, list_IDs, base_dir, cut=64600):
        """
        Args:
            list_IDs: List of file IDs (without extension)
            base_dir: Base directory containing audio files
            cut: Audio length to use (default: 64600 samples = ~4 sec at 16kHz)
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]

        # Load audio file
        audio_path = os.path.join(self.base_dir, f'{key}.flac')
        X, fs = librosa.load(audio_path, sr=16000)

        # Pad to fixed length
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)

        return x_inp, key


def get_class_weights(labels_dict):
    """
    Calculate class weights for imbalanced dataset.

    Args:
        labels_dict: Dictionary mapping file_id -> label

    Returns:
        torch.FloatTensor of class weights [weight_spoof, weight_bonafide]
    """
    labels = list(labels_dict.values())
    n_bonafide = sum(labels)
    n_spoof = len(labels) - n_bonafide
    n_total = len(labels)

    # Inverse frequency weighting
    weight_spoof = n_total / (2 * n_spoof) if n_spoof > 0 else 1.0
    weight_bonafide = n_total / (2 * n_bonafide) if n_bonafide > 0 else 1.0

    # Normalize
    total_weight = weight_spoof + weight_bonafide
    weight_spoof /= total_weight
    weight_bonafide /= total_weight

    print(f'Class distribution: Bonafide={n_bonafide}, Spoof={n_spoof}')
    print(f'Class weights: [Spoof={weight_spoof:.4f}, Bonafide={weight_bonafide:.4f}]')

    return torch.FloatTensor([weight_spoof, weight_bonafide])

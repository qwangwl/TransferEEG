# -*- encoding: utf-8 -*-
'''
file       :seed_feature.py
Date       :2025/02/12 10:59:16
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import re
import numpy as np
import scipy.io as scio
from pathlib import Path
from typing import List, Tuple, Dict, Union

class SEEDFeatureDataset(object):
    """SEED EEG Feature Dataset Loader

    Provides feature loading functionality for the SEED dataset, supporting data selection by channels, subjects, and sessions.
    Required dataset directory structure:
    - root_path/
        - 1/
        - - 1_20131027.mat
        - - ...
        - 2/
        - - 1_20131030.mat
        - - ...
        - 3/
        - - 1_20131107.mat
        - - ...
        - label.mat
    Args:
        root_path (str): Root path of the dataset (default: ".\\ExtractedFeatures")
        feature_name (str): Name of the feature to extract (default: "de_LDS")
        channels (List[str]): List of selected EEG channels, None means all (default: None)
        subjects (List[int]): List of selected subject IDs, None means all (default: None)
        sessions (Union[List[int], int]): Selected session IDs, None means all (default: [1])

    Info:
        Since cross-session experiments are common in SEED, we set subjects and sessions for flexible experimental settings.
    """

    CHANNELS_LIST = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
        'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]

    def __init__(
        self,
        root_path: str = "./ExtractedFeatures",
        feature_name: str = "de_LDS",
        channels: List[str] = None,
        subjects: List[int] = None,
        sessions: Union[List[int], int] = [1]
    ):
        super(SEEDFeatureDataset, self).__init__()
        self.root_path = Path(root_path)
        self.feature_name = feature_name
        self.channel_indices = self._get_channel_indices(channels)
        
        # The labels are stored in a separate file, so they need to be loaded separately
        self.trials_labels = self._load_trial_labels()

        meta_info = self._get_meta_info(sessions, subjects)
        self._dataset_cache = self._process_all_subjects(meta_info)

    def get_dataset(self):
        return self._dataset_cache
    
    def get_feature_dim(self):
        return self._dataset_cache["data"].shape[-1]
    
    def _process_all_subjects(self, meta_info: List[Dict]):
        """Load EEG data for the given session and subject information"""
        data, labels, groups = [], [], []
        for info in meta_info:
            samples = self._process_one_subject(info)
            data.append(samples["data"])
            labels.append(samples["labels"])
            groups.append(samples["groups"])
        return {
            "data": np.concatenate(data),
            "labels": np.concatenate(labels),
            "groups": np.concatenate(groups)
        }


    def _process_one_subject(self, subject_info: Dict):
        # Load .mat file containing the EEG data
        mat_data = scio.loadmat(subject_info["file_path"], verify_compressed_data_integrity=False)

        # Extract trial IDs from keys that start with "de_LDS"
        trial_ids = [int(re.findall(r"de_LDS(\d+)", key)[0]) for key in mat_data.keys() if key.startswith("de_LDS")]

        # Initialize lists to store EEG data, group information, and labels
        data, groups, labels = [], [], []

        for trial_id in trial_ids:
            # Extract the EEG data for the current trial, transpose dimensions and select channels
            trial_data = mat_data[f"{self.feature_name}{trial_id}"].transpose(1, 0, 2)[:, self.channel_indices]
            num_samples = trial_data.shape[0]
            # Create group information (trial_id, subject_id, session_id)
            trial_group = np.hstack([
                np.ones((num_samples, 1), dtype=np.int16) * subject_info["subject"],
                np.ones((num_samples, 1), dtype=np.int16) * trial_id,
                np.ones((num_samples, 1), dtype=np.int16) * subject_info["session"]
            ])
            # Assign label based on the session and trial ID
            trial_label = np.full(shape=num_samples, fill_value=self.trials_labels[trial_id - 1], dtype=np.int16)

            # Append data, group info, and labels for the current trial
            data.append(trial_data)
            groups.append(trial_group)
            labels.append(trial_label)

        # Return a dictionary containing the concatenated data, labels, and group info
        return {
            "data": np.concatenate(data),
            "labels": np.concatenate(labels),
            "groups": np.concatenate(groups)
        }
    
    def _load_trial_labels(self) -> np.ndarray:
        # Load the label file which contains trial labels
        label_path = self.root_path / "label.mat"
        return scio.loadmat(label_path,
                            verify_compressed_data_integrity=False)['label'][0]


    def _get_meta_info(self, sessions: Union[List[int], int], subjects: List[int]) -> List[Dict]:
        """get meta information of the dataset"""
        sessions = [sessions] if isinstance(sessions, int) else sessions or [1, 2, 3]
        subjects = subjects or list(range(1, 16))

        meta_info = []
        for session_id in sessions:
            session_path = self.root_path / str(session_id)
            subject_mat_files = [
                file for file in session_path.glob("*.mat") 
                if self._parse_subject_id(file) in subjects
            ]

            for file_path in subject_mat_files:
                meta_info.append({
                    "session": session_id,
                    "subject": self._parse_subject_id(file_path),
                    "file_path": file_path
                })

        return meta_info
    
    @staticmethod
    def _parse_subject_id(file_path: Path) -> int:
        """parser subject ID from file name"""
        return int(file_path.stem.split("_")[0])

    def _get_channel_indices(self, channels: List[str]) -> np.ndarray:
        """get indices of the specified channels"""
        if not channels:
            return np.arange(len(self.CHANNELS_LIST))
            
        invalid_channels = set(channels) - set(self.CHANNELS_LIST)
        if invalid_channels:
            raise ValueError(f"invalid_channels: {invalid_channels}")
            
        return np.where(np.isin(self.CHANNELS_LIST, channels))[0]
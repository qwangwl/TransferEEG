# -*- encoding: utf-8 -*-
'''
file       :seediv_feature.py
Date       :2025/02/12 11:26:49
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import re
import numpy as np
import scipy.io as scio
from pathlib import Path
from typing import List, Tuple, Dict, Union
from datasets import SEEDFeatureDataset

class SEEDIVFeatureDataset(SEEDFeatureDataset):
    """
    SEED-IV dataset. Loads feature data and supports filtering by channels, subjects, and sessions.
    Required dataset directory structure:
    - root_path/
        - 1/
            - 1_20160518.mat
            - ...
        - 2/
            - 1_20161125.mat
            - ...
        - 3/
            - 1_20161126.mat
            - ...
    
    Args:
        root_path (str): Root path of the dataset (default: ".\\eeg_feature_smooth")
        feature_name (str): Name of the feature to extract (default: "de_LDS")
        channels (List[str]): List of selected EEG channels, None for all (default: None)
        subjects (List[int]): List of selected subject IDs, None for all (default: None)
        sessions (Union[List[int], int]): Selected session IDs, None for all (default: [1])
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
        root_path: str = ".\\eeg_feature_smooth",
        feature_name: str = "de_LDS",
        channels: List[str] = None,
        subjects: List[int] = None,
        sessions: Union[List[int], int] = [1]
    ):
        super(SEEDIVFeatureDataset, self).__init__(
            root_path,
            feature_name, 
            channels,
            subjects,
            sessions
        )

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

            trial_label = np.full(shape=num_samples, fill_value=self.trials_labels[subject_info["session"] - 1][trial_id - 1], dtype=np.int16)

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
        """获取标签信息"""
        return np.array([
            [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
            [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
        ])

# -*- encoding: utf-8 -*-
'''
file       :dreamer_feature.py
Date       :2025/02/12 20:05:48
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import numpy as np
import pickle
from pathlib import Path
import scipy.io as scio
from typing import List, Tuple, Dict, Union, Optional
from datasets.base import SignalDataset

class DreamerDataset(SignalDataset):

    CHANNELS_LIST = [
        'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4',
        'F8', 'AF4'
    ]
    LABELS_LIST = ['valence', 'arousal', 'dominance']
    EEG_SAMPLING_RATE = 128 
    
    def __init__(
            self,
            root_path: str = ".\\dreamer",
            channels: List[str] = None,
            labels: Union[str, List[str]] = None,
            window_sec: int = 1,
            step_sec: Optional[float] = None,
            feature_name: str = None,
            **kwargs):
        
        super(DreamerDataset, self).__init__(
            root_path=root_path,
            channels=channels,
            labels=labels,
            window_sec=window_sec,
            step_sec=step_sec,
            feature_name=feature_name,
            **kwargs
        )
        self.num_of_subjects = 23
        self._baseline_cache, self._stimulus_cache = \
            self._process_all_subjects()

    def get_baseline(self):
        return self._baseline_cache
    
    def get_stimulus(self):
        return self._stimulus_cache
    
    def get_dataset(self):
        changed_data = self._subtract_baseline(
            self._stimulus_cache["data"], self._stimulus_cache["groups"],  
            self._baseline_cache["data"], self._baseline_cache["groups"])
        # changed_data = self._stimulus_cache["data"]
        return {
            "data" : changed_data,
            "labels" : self._stimulus_cache["labels"],
            "groups" : self._stimulus_cache["groups"]
        }

    def get_feature_dim(self):
        return self._stimulus_cache["data"].shape[-1]

    def _process_all_subjects(self):
        baseline_data_list = []  # List to hold all baseline feature data
        baseline_group_list = []  # List to hold all baseline group data
        stimulus_data_list = []  # List to hold all stimulus feature data
        stimulus_group_list = []  # List to hold all stimulus group data
        stimulus_labels_list = []  # List to hold all stimulus labels data
        
        for info in self._get_meta_info():
            # Read data for each subject
            mat_data = self._load_file(self.root_path / "DREAMER.mat")

            # Process the data for this subject
            baseline, stimulus = self._process_one_subject(info["subject"], mat_data)

            # Append the data to the respective lists
            baseline_data_list.append(baseline["data"])
            baseline_group_list.append(baseline["groups"])

            stimulus_data_list.append(stimulus["data"])
            stimulus_group_list.append(stimulus["groups"])
            stimulus_labels_list.append(stimulus["labels"])

        # Once all subjects are processed, convert lists to numpy arrays
        _baseline_cache = {
            "data": np.vstack(baseline_data_list),  # Stack all baseline feature data
            "groups": np.vstack(baseline_group_list)  # Stack all baseline group data
        }
        _stimulus_cache = {
            "data": np.vstack(stimulus_data_list),  # Stack all stimulus feature data
            "groups": np.vstack(stimulus_group_list),  # Stack all stimulus group data
            "labels": np.vstack(stimulus_labels_list)  # Concatenate all stimulus labels
        }
        return _baseline_cache, _stimulus_cache
    
    def _process_one_subject(
        self, 
        subject_id: int,
        mat_data: np.ndarray,
    ):
        # Since the number of trials is different, we still need to process them one by one.

        num_of_trials = mat_data["DREAMER"][0, 0]["Data"][0, subject_id-1]["ScoreValence"][0, 0].shape[0]

        baseline_data_list = []
        stimulus_data_list = []
        stimulus_labels_list = []
        baseline_group_list = []
        stimulus_group_list = []

        for trial_id in range(num_of_trials):
            # Process baseline data
            baseline_data = mat_data['DREAMER'][0, 0]['Data'][0, subject_id-1]['EEG'][0, 0]['baseline'][0, 0][trial_id, 0]
            baseline_data = np.swapaxes(baseline_data, 0, 1)
            baseline_data = np.expand_dims(baseline_data, axis=0)

            # Process stimulus data
            stimulus_data = mat_data['DREAMER'][0, 0]['Data'][0, subject_id-1]['EEG'][0, 0]['stimuli'][0, 0][trial_id, 0]
            stimulus_data = np.swapaxes(stimulus_data, 0, 1)
            stimulus_data = np.expand_dims(stimulus_data, axis=0)

            # Current trial labels
            labels = np.array([
                mat_data['DREAMER'][0, 0]['Data'][0, subject_id-1]['ScoreValence'][0, 0][trial_id, 0],
                mat_data['DREAMER'][0, 0]['Data'][0, subject_id-1]['ScoreArousal'][0, 0][trial_id, 0],
                mat_data['DREAMER'][0, 0]['Data'][0, subject_id-1]['ScoreDominance'][0, 0][trial_id, 0]
            ])[np.newaxis, :]

            baseline_data = baseline_data[:, self.channel_indices, :]
            stimulus_data = stimulus_data[:, self.channel_indices, :]
            labels = labels[:, self.label_indices]

            # print(baseline_data.shape, stimulus_data.shape)
            # Segment baseline and stimuli data
            baseline_group, baseline_data, _ = self._segment_signal(baseline_data, None)
            stimulus_group, stimulus_data, stimulus_labels = self._segment_signal(stimulus_data, labels)
            # print(stimulus_group)

            baseline_feature = self.feature_extractor(baseline_data)
            stimulus_feature = self.feature_extractor(stimulus_data)

            baseline_group = np.column_stack((np.full_like(baseline_group, subject_id), baseline_group * (trial_id + 1)))
            stimulus_group = np.column_stack((np.full_like(stimulus_group, subject_id), stimulus_group * (trial_id + 1)))

            # Append the results for this trial
            baseline_group_list.append(baseline_group)
            baseline_data_list.append(baseline_feature)
            stimulus_group_list.append(stimulus_group)
            stimulus_data_list.append(stimulus_feature)
            stimulus_labels_list.append(stimulus_labels)

        # Concatenate all trials at once
        all_baseline_group = np.concatenate(baseline_group_list, axis=0)
        all_baseline_data = np.concatenate(baseline_data_list, axis=0)
        all_stimulus_group = np.concatenate(stimulus_group_list, axis=0)
        all_stimulus_data = np.concatenate(stimulus_data_list, axis=0)
        all_stimulus_labels = np.concatenate(stimulus_labels_list, axis=0)

        baseline = {
            "data": all_baseline_data,
            "groups": all_baseline_group
        }

        stimulus = {
            "data": all_stimulus_data,
            "groups": all_stimulus_group,
            "labels": all_stimulus_labels
        }
        return baseline, stimulus

    def _load_file(self, file_path: Path):
        mat_data = scio.loadmat(file_path,
                        verify_compressed_data_integrity=False)
        return mat_data
    
    def _get_meta_info(self) -> List[Dict]:
        meta_info = []
        for subject in list(range(1, self.num_of_subjects+1)):
            meta_info.append(
                {"subject": subject}
            )
        return meta_info


# -*- encoding: utf-8 -*-
'''
file       :seediv_feature.py
Date       :2024/11/12 21:45:59
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

# 对SEEDIV数据集中 特征的读取。
import re
import os
import numpy as np
import scipy.io as scio

class SEEDIVFeatureDataset(object):
    """
        使用这个Dataset, 必须要下载SEEDIV数据集, 其数据集内部的存储路径为:
        - 1 
        - - 1_20160518.mat
        - - ...
        - - 15_20150508.mat
        - 2 
        - - 1_20161125.mat
        - - ...
        - 3
        - - 1_20161126.mat
        - - ...
        
        该路径包含三个文件夹, 每一个文件夹表示一个session的数据, 文件夹内部包含当前session进行的15个受试者的数据
    
    Args:
        root_path(str): 下载的SEEDIV文件,（default: ".\\eeg_feature_smooth" 
        feature(str): 所需要的特征, (default:, "de_LDS") 
        channels(list, None): 所需要的脑电通道列表, 默认为None, 为None时表示获得所有62个脑电通道, 通道列表的名称详见: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html (default: None)
        subjects(list, None): 所需要的受试者数据, 默认为None, 表示所有的受试者 (default: None)
        session(list, None): 所需要的session数据, 设置为None时, 获取所有session数据 (default: [1])
    """
    def __init__(self, 
                 root_path: str = ".\\eeg_feature_smooth",
                 feature: str = "de_LDS",
                 channels: list | None = None,
                 subjects: list | None = None,
                 session: list | int | None = [1],
                 ):
        super(SEEDIVFeatureDataset, self).__init__()

        channel_index = self._get_channel_index(channels)
        meta_info = self._process_record(root_path, session, subjects)
        self.eeg = self._read_data(meta_info, channel_index, root_path, feature)

    def data(self):
        return self.eeg

    def _read_data(self, meta_info, channel_index, root_path, feature):
        
        Data = []
        Label = []
        Group = []
        
        for info in meta_info: 
            samples = self._sampleIO(root_path, info, channel_index, feature)
            Data.append(samples[0])
            Label.append(samples[1])
            Group.append(samples[2])
        
        Data = np.concatenate(Data)
        Label = np.concatenate(Label)
        Group = np.concatenate(Group)
        
        return Data, Label, Group
    

    def _sampleIO(self, root_path, info, channel_index, feature):
        # print(info)
        # 读取数据
        eeg = scio.loadmat(os.path.join(root_path, str(info["session"]), info["file_name"]),
                           verify_compressed_data_integrity=False)
        
        # print(eeg.keys())
        
        labels = [
            [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
            [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
        ]
        
        trial_ids = [int(re.findall(r"de_LDS(\d+)", key)[0]) for key in eeg.keys() if key.startswith("de_LDS")]

        data = []
        group = []
        label = []
        
        # 获取每一个trial的数据
        for trial_id in trial_ids:
            # 数据
            eeg_trial = eeg[feature + str(trial_id)].transpose(1, 0, 2)[:, channel_index]
            
            num_trial = eeg_trial.shape[0]
            
            # 保存当前trial的信息，如第几个session，第几个subject和第几个trial
            group_trial_id = np.ones((num_trial, 1), dtype=np.int16) * trial_id
            group_subject_id = np.ones((num_trial, 1), dtype=np.int16) * info["subject"]
            group_session_id = np.ones((num_trial, 1), dtype=np.int16) * info["session"]
            # 将group拼接
            group_trial = np.hstack((group_session_id, group_subject_id, group_trial_id))
            
            # 将标签对齐。
            label_trial = np.full(shape = num_trial, fill_value = labels[info['session']-1][trial_id - 1], dtype=np.int16)

            data.append(eeg_trial)
            group.append(group_trial)
            label.append(label_trial)
        
        data = np.concatenate(data)
        group = np.concatenate(group)
        label = np.concatenate(label)
        
        return data, label, group


    def _get_channel_index(self, channels):
        # 获取所提取的脑电通道索引
        SEEDIV_CHANNEL_LIST = np.array([
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
            'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
            'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
            'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ])
        
        if channels is None:
            index_of_channel = np.arange(len(SEEDIV_CHANNEL_LIST))
        
        elif isinstance(channels, list):
            channels = np.array(channels)
            
            err_chan = np.setdiff1d(channels, SEEDIV_CHANNEL_LIST)
            if err_chan.size:
                raise ValueError("No {} channels".format(list(err_chan)))
            
            index_of_channel = np.in1d(SEEDIV_CHANNEL_LIST, channels).nonzero()[0]
            
        return index_of_channel
    
    def _process_record(self, root_path, session, subjects):
        # 获取所需要数据的文件名
        if isinstance(session, int):
            session = [session]
        elif session is None:
            session = [1, 2, 3]
        if subjects is None:
            subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
        meta_info = []
        for sess in session:
            file_list = os.listdir(os.path.join(root_path, str(sess)))
            
            subject_file_list = [
                file_name for file_name in file_list
                if int(file_name.split("_")[0]) in subjects
            ]
            
            for file_name in subject_file_list:
                sub_info = {
                    "session" : sess,
                    "subject" : int(file_name.split("_")[0]),
                    "file_name" : file_name
                }
                meta_info.append(sub_info)

        return meta_info     

if __name__ == "__main__":
    path = "E:\\EEG_DataSets\\SEED_IV\eeg_feature_smooth"
    # seed = SEEDFeatureDataset(root_path = path, channels=["FP1", "F5", "F8"])
    seed = SEEDIVFeatureDataset(root_path = path, channels=["FP1", "F5", "F8"], session=[1, 2, 3]).data()
    print(seed[0].shape, seed[1].shape, seed[2].shape)
    print(np.unique(seed[1]))
    print(np.unique(seed[2][:, 1]))
    print(np.unique(seed[2][:, 2]))
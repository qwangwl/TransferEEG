# -*- encoding: utf-8 -*-
'''
file       :psd.py
Date       :2025/02/12 17:02:01
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import numpy as np
from scipy import signal
from typing import Tuple

class PSD:
    def __init__(
            self, 
            sf: int = 128, 
            nperseg: int = 128, 
            band: Tuple[int, ...] = (1, 4, 8, 13, 31, 50)):
        """
        Initializes the PSD class for computing power spectral density (PSD) features.

        Parameters:
        -----------
        sf : int, optional
            Sampling frequency of the EEG data (in Hz). Default is 128 Hz.

        nperseg : int, optional
            The length of each segment used in Welch's method. Default is 128 samples.

        band : Tuple[int, ...], optional
            A tuple of frequency band boundaries. Default is (1, 4, 8, 13, 31, 50).
            These represent common EEG frequency bands: Delta, Theta, Alpha, Beta, Gamma.
        """
        self.sf = sf
        self.nperseg = nperseg
        self.band = band

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Makes the PSDFeatureExtractor callable. This method allows the class instance 
        to be called as a function to extract PSD features from EEG data.

        Parameters:
        -----------
        data : np.ndarray
            Input EEG data of shape (trials, channels, samples).
        """
        return self._compute_psd(data)

    def feature_dim(self):
        """
        Returns the number of features extracted by the PSD method.
        """
        return len(self.band) - 1

    def _compute_psd(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the Power Spectral Density (PSD) using Welch's method.

        Parameters:
        -----------
        data : np.ndarray
            Input EEG data. This should be a 2D or 3D array where the shape can be either:
            (n_samples x n_points) or (n_samples x n_channels x n_points)
        """
        # Convert frequency band boundaries into a NumPy array for easier handling
        band = np.array(self.band)

        # Calculate the power spectral density using Welch's method
        freqs, power = signal.welch(data, fs=self.sf, nperseg=self.nperseg)

        # Split the frequency axis based on the defined frequency bands, excluding the first and last frequencies (boundary bands)
        freq_bands = np.hsplit(freqs, band)[1:-1]  # Remove first and last frequencies (boundary)

        # For each frequency band, find the indices corresponding to the frequencies within that band
        pindex = [np.where(np.in1d(freqs, fb))[0] for fb in freq_bands]

        # For each band, compute the mean power across the trials, slices, and channels
        features = np.stack([np.mean(power[..., idx], axis=-1) for idx in pindex], axis=-1)

        return features

    


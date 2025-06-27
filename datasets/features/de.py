# -*- encoding: utf-8 -*-
'''
file       :de.py
Date       :2025/02/12 17:11:47
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import numpy as np
import math
from scipy.signal import butter, lfilter

class DE(object):
    """
    Initializes the DE class for computing differential entropy(DE) features.

    Parameters:
    -----------
    fs : int, optional
        Sampling frequency in Hz. Default is 128 Hz.
    band : tuple, optional
        A tuple of frequency band limits. Default is (1, 4, 8, 13, 30, 50).
        These represent common EEG frequency bands: Delta, Theta, Alpha, Beta, Gamma.
    order : int, optional
        Order of the bandpass filter. Default is 3.
    """
    def __init__(
            self,
            fs: int = 128, 
            order: int = 3,
            band: tuple[int, ...] = (1, 4, 8, 13, 31, 50)):
        super(DE, self).__init__()
        self.fs = fs
        self.band = band
        self.order = order

    def __call__(self, data):
        """
        Makes the EEGFeatureExtractor callable. This method allows the class instance 
        to be called as a function to extract features from EEG data.
        """
        return self._extract_differential_entropy(data)
    
    def feature_dim(self):
        """
        Returns the number of features extracted by the DE method.
        """
        return len(self.band) - 1

    def _design_bandpass_filter(self, lowcut, highcut):
        """
        Designs a Butterworth bandpass filter.
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a
    
    def _apply_bandpass_filter(self, data, lowcut, highcut):
        """
        Applies a Butterworth bandpass filter to the input signal.
        """
        b, a = self._design_bandpass_filter(lowcut, highcut)
        y = lfilter(b, a, data)
        return y
    
    def _compute_variance_entropy(self, signal):
        """
        Computes the Variance-based Differential Entropy (DE) of a signal.

        The DE is computed as the logarithm of the variance of the signal.
        """
        variance = np.var(signal, ddof=1, axis=-1)
        de = 0.5 * np.log(2 * math.pi * math.e * variance)
        return de

    def _extract_differential_entropy(self, data):
        """
        Computes the Differential Entropy (DE) for each frequency band in the given data.

        This function first applies bandpass filters to the input signal, splitting the signal 
        into the frequency bands specified in the `band` attribute. Then, it calculates the DE 
        for each band.
        """
        n_trials, n_channels, _ = data.shape
        n_bands = len(self.band) - 1
        features = np.empty((n_trials, n_channels, n_bands))

        # Apply bandpass filter for each band and compute DE
        for b in range(n_bands):
            lowcut = self.band[b]
            highcut = self.band[b + 1]
            # Apply bandpass filter
            filtered_data = self._apply_bandpass_filter(data, lowcut, highcut)
            # Compute DE for the filtered data
            de = self._compute_variance_entropy(filtered_data)
            # Store the DE features for this band
            features[:, :, b] = de

        return features
    
    



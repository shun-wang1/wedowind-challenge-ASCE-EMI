# -*- coding: utf-8 -*-

import numpy as np
epsn = 1e-8

def rms_fea(a):
	return np.sqrt(np.mean(np.square(a)))

def am_fea(a):
	return np.mean(np.abs(a))

def max_fea(a):
	return np.max(np.abs(a))

def pp_fea(a):
	return np.max(a)-np.min(a)

def shape_factor(a):
	return rms_fea(a)/(am_fea(a)+epsn)

def peak_factor(a):
	return max_fea(a)/(rms_fea(a)+epsn)

def impluse_factor(a):
	return max_fea(a)/(am_fea(a)+epsn)

def crest_factor(signal: np.ndarray) -> float:
	"""Calculate Crest Factor (peak value / RMS)"""
	rms = np.sqrt(np.mean(np.square(signal)))
	peak = np.max(np.abs(signal))
	return peak / rms

def clearance_factor(signal: np.ndarray) -> float:
	"""Calculate Clearance Factor (peak value / square of mean of square root)"""
	mean_sqrt = np.mean(np.sqrt(np.abs(signal)))
	peak = np.max(np.abs(signal))
	return peak / (mean_sqrt ** 2)

def fft_fft(sequence_data):
	fft_trans = np.abs(np.fft.fft(sequence_data))
	# dc = fft_trans[0]
	freq_spectrum = fft_trans[1:int(np.floor(len(sequence_data) * 1.0 / 2)) + 1]
	_freq_sum_ = np.sum(freq_spectrum)
	return freq_spectrum, _freq_sum_

def fft_mean(sequence_data):
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	return np.mean(freq_spectrum)

def fft_var(sequence_data):
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	return np.var(freq_spectrum)

def fft_std(sequence_data):
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	return np.std(freq_spectrum)

def fft_entropy(sequence_data):
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	if _freq_sum_ == 0:
		return 0
	pr_freq = freq_spectrum * 1.0 / _freq_sum_
	entropy = -1 * np.sum([np.log2(p+epsn) * p for p in pr_freq])
	num_bins = len(pr_freq)
	max_entropy = np.log2(num_bins)
	normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
	return normalized_entropy

def fft_energy(sequence_data):
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	return np.sum(freq_spectrum ** 2) / len(freq_spectrum)

def fft_skew(sequence_data):
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	_fft_mean, _fft_std = fft_mean(sequence_data), fft_std(sequence_data)
	return np.mean([0 if _fft_std < epsn else np.power((x - _fft_mean) / _fft_std, 3)
					for x in freq_spectrum])

def fft_kurt(sequence_data):
	freq_spectrum, _freq_sum_ = fft_fft(sequence_data)
	_fft_mean, _fft_std = fft_mean(sequence_data), fft_std(sequence_data)
	SK_value = np.mean([0 if _fft_std < epsn else np.power((x - _fft_mean) / _fft_std, 4)
					for x in freq_spectrum]) - 3
	return SK_value

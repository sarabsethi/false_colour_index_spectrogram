#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core IO, DSP and utility functions."""

import os
import six

import audioread
import numpy as np
import scipy.signal
import scipy.fftpack as fft
import resampy

from librosa.core.time_frequency import frames_to_samples, time_to_samples
from librosa import cache
from librosa import util
from librosa.util.exceptions import ParameterError

# Resampling bandwidths as percentage of Nyquist
BW_BEST = resampy.filters.get_filter('kaiser_best')[2]
BW_FASTEST = resampy.filters.get_filter('kaiser_fast')[2]


# -- CORE ROUTINES --#
# Load should never be cached, since we cannot verify that the contents of
# 'path' are unchanged across calls.
def load_yield_chunks(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32, res_type='kaiser_best', choplenspls=0):
    """Load an audio file as a floating point time series.
    This is MODIFIED from librosa's own load() function, to yield chunks one-by-one so they never all need to be loaded into memory.

    Parameters
    ----------
    path : string
        path to the input file.

        Any format supported by `audioread` will work.

    sr   : number > 0 [scalar]
        target sampling rate

        'None' uses the native sampling rate

    mono : bool
        convert signal to mono

    offset : float
        start reading after this time (in seconds)

    duration : float
        only load up to this much audio (in seconds)

    dtype : numeric type
        data type of `y`

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').

            To use a faster method, set `res_type='kaiser_fast'`.

            To use `scipy.signal.resample`, set `res_type='scipy'`.

    choplenspls : int
        number of samples in each chunk to be yielded.

    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]
        audio time series

    sr   : number > 0 [scalar]
        sampling rate of `y`


    Examples
    --------
    >>> # Load a wav file
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename)
    >>> y
    array([ -4.756e-06,  -6.020e-06, ...,  -1.040e-06,   0.000e+00], dtype=float32)
    >>> sr
    22050

    >>> # Load a wav file and resample to 11 KHz
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename, sr=11025)
    >>> y
    array([ -2.077e-06,  -2.928e-06, ...,  -4.395e-06,   0.000e+00], dtype=float32)
    >>> sr
    11025

    >>> # Load 5 seconds of a wav file, starting 15 seconds in
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    >>> y
    array([ 0.069,  0.1  , ..., -0.101,  0.   ], dtype=float32)
    >>> sr
    22050

    """

    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = to_mono(y)

        if sr is not None:
            y = resample(y, sr_native, sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)



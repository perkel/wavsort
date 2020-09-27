#!/usr/bin/env python

"""Class to make playing around with wav files easy and fun.

See code at bottom for example usage.

Written Feb 2016 by Brandon Warren (bwarren@uw.edu)

"""

import wave
import numpy
import matplotlib.pyplot as plt
from matplotlib import mlab
from pylab import gca

def specgram_freq_limit(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
             window=mlab.window_hanning, noverlap=128,
             cmap=None, xextent=None, pad_to=None, sides='default',
             scale_by_freq=None, minfreq=0.0, maxfreq=None, ax=None, **kwargs):
    """Wrapper for matplotlib.pyplot.specgram to remove frequencies not of
    interest. You can call set_ylim([0.0, max_freq]), but the colors will
    remain unchanged. This will cause the full range of color to appear.

    From http://stackoverflow.com/questions/19468923/cutting-of-unused-frequencies-in-specgram-matplotlib
    """

    if ax is None:
        ax = gca()
    Pxx, freqs, bins = mlab.specgram(x, NFFT=NFFT, Fs=Fs, detrend=detrend,
         window=window, noverlap=noverlap, pad_to=pad_to, sides=sides,
        scale_by_freq=scale_by_freq, **kwargs)

    # modified here
    #####################################
    if maxfreq is not None:
        Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        freqs = freqs[(freqs >= minfreq) & (freqs <= maxfreq)]
    #####################################

    Z = 10. * numpy.log10(Pxx)
    Z = numpy.flipud(Z)

    if xextent is None: xextent = 0, numpy.amax(bins)
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    im = ax.imshow(Z, cmap, extent=extent, **kwargs)
    ax.axis('auto')

    return Pxx, freqs, bins, im

class SignalLab(object):
    """Class for opening, plotting, and analyzing sound files (only wav for now).

    Attributes:
        path (str): full path of opened file
        sample_rate (float) : sample rate of opened file in Hz
        n_wav_samps (int): number of samples
        sound_data (numpy int16 array): the raw sound data
        delta_t (float): 1/sample_rate
        sample_times (numpy float array): time of each sample, first time is zero
        max_pitch_freq (float): based on user-supplied max_pitch_freq
    """

    def __init__(self, path, max_pitch_freq=4.2e3):
        """This just opens, reads, and closes the wav file.

        Args:
            path (str): full path of file to open
            max_pitch_freq (float): used by cepstrum. Max pitch freq we care about.
        """
        try:
            snd = wave.open(path, 'r')
            self.path = path
            if snd.getsampwidth() != 2:
                raise TypeError('{} has {}-byte samples. Expecting 2-byte samples.')
            self.sample_rate = float(snd.getframerate())
            self.n_wav_samps = snd.getnframes()
            stream = snd.readframes(self.n_wav_samps)
            self.sound_data = numpy.fromstring(string=stream, dtype=numpy.int16)
            self.delta_t = 1.0/self.sample_rate
            self.sample_times = numpy.arange(0,
                                             (self.n_wav_samps+1)*self.delta_t,
                                             self.delta_t)
            self.window = numpy.zeros(2) # don't know window size. won't be 2.

            # calc self._n_cepstrum_points_to_skip_for_pitch - keep 1st point above
            # max_pitch_freq and all the ones below.
            # actual max freq is stored in self.max_pitch_freq
            # e.g. sample rate is 44.1 kHz, we want to see 20kHz, so skip
            # first 3 points of cepstrum
            #cepstrum_[0] = 0 # no shift MUST CLEAR THIS, zero shift is always max
            #cepstrum_[1] = 0 # 44.1 kHz (won't be at sample rate, or even 1/2 it
            #cepstrum_[2] = 0 # 22 kHz - don't want this often-large value boosting goodness-of-pitch
            self._n_cepstrum_points_to_skip_for_pitch = 1 # always clear 1st point (zero shift)
            while True:
                next_f = self.sample_rate/(self._n_cepstrum_points_to_skip_for_pitch+1)
                if next_f < max_pitch_freq:
                    break
                self._n_cepstrum_points_to_skip_for_pitch += 1
            self.max_pitch_freq = self.sample_rate/self._n_cepstrum_points_to_skip_for_pitch
        finally:
            snd.close()

    def _plot_time(self, data, offset_i, num_points, title='', ylabel='Counts'):
        """Used internally to plot time history data.

        Args:
            data (numpy array): data to plot. Will use first sample passed, so
                    caller must send slice if he wants to plot from an offset.
            offset_i (int): 0-based index, used to index into sample_times
            num_points (int): number of samples to plot
            title (str): plot title
            ylabel (str): label to use for Y-axis

        May be overridden to plot using different library or method (e.g. to PNG file).
        """
        plt.figure(figsize=(8.0, 4.0), dpi=80)
        subplot = plt.subplot(1,1,1) # only needed if I want to call subplot methods
        plt.plot(self.sample_times[offset_i:offset_i+num_points],
                 data[:num_points],
                 'r')
        plt.title(title)
        plt.xlabel('Seconds')
        plt.ylabel(ylabel)
        subplot.set_autoscale_on(True)
        plt.grid(True)

    def _plot_freq(self, data, blocksize, title='', xlabel='Hz', ylabel=''):
        plt.figure(figsize=(8.0, 4.0), dpi=80)
        subplot = plt.subplot(1,1,1) # only needed if I want to call subplot methods
        delta_f = self.sample_rate/blocksize # 1/T
        x_scale = numpy.arange(0, (blocksize/2+1)*delta_f, delta_f)[:blocksize/2]
        plt.plot(x_scale, data, 'r')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        subplot.set_autoscale_on(True)
        plt.grid(True)

    def plot_time(self, offset_time=0.0, duration=None, num_points=None, 
                  title=None):
        """Plot time history of sound file.

        Args:
            offset_time (float): offset from start of file to begin, in seconds
            duration (float): number of seconds to plot
            num_points (int): num samples to plot (if duration not specified)
            title (str): title of plot
        """
        if title is None:
            title = self.path
        offset_i = int(0.5 + offset_time*self.sample_rate)
        if duration:
            num_points = int(0.5 + duration*self.sample_rate)
        elif num_points is None:
            num_points = self.n_wav_samps
        max_num_points = self.n_wav_samps - offset_i
        num_points = min(num_points, max_num_points)
        self._plot_time(self.sound_data[offset_i:], offset_i, num_points, title=title)

    def plot_spectrogram(self, offset_time=0.0, duration=None, num_points=None,
                  blocksize=512, max_freq=None, title=None):
        """Plot spectrogram of sound file.

        Args:
            offset_time (float): offset from start of file to begin, in seconds
            duration (float): number of seconds to plot
            num_points (int): num samples to plot (if duration not specified)
            blocksize (int): FFT blocksize
            max_freq (float): max freq of interest
            title (str): title of plot
        """
        if title is None:
            title = self.path
        if offset_time:
            title += ' offset of {0:.3f} sec'.format(offset_time)
        offset_i = int(0.5 + offset_time*self.sample_rate)
        if duration:
            num_points = int(0.5 + duration*self.sample_rate)
        elif num_points is None:
            num_points = self.n_wav_samps
        max_num_points = self.n_wav_samps - offset_i
        num_points = min(num_points, max_num_points)

        plt.figure(figsize=(8.0, 4.0), dpi=80)
        #spectrum, freqs, t, im  = plt.specgram(
        spectrum, freqs, t, im  = specgram_freq_limit(
            self.sound_data[offset_i:offset_i+num_points], NFFT=blocksize,
            Fs=self.sample_rate,
            window=mlab.window_hanning, noverlap=blocksize/2,
            maxfreq=max_freq) # new - limit freq
        plt.title(title)
        plt.xlabel('Seconds')
        plt.ylabel('Hz')

    def power_spectrum(self, offset_time, blocksize, window_it=True,
                       plot_it=True, title=None):
        """Compute and plot power spectrum of sound file.
        """
        # TODO: verify alg
        i = int(0.5 + offset_time*self.sample_rate)
        fft_input = self.sound_data[i:i+blocksize] # numpy - does not copy
        if window_it:
            if self.window.shape != (blocksize,):
                # hann window
                d = 2.0 * numpy.pi / (blocksize-1)
                twopi = 2.0 * numpy.pi
                self.window = 0.5 * (1.0 - numpy.cos(
                    numpy.arange(0.0, twopi+d, d, dtype=numpy.float32)))
            fft_input = fft_input.copy() # make copy            fft_input *= self.window
        fft = numpy.fft.fft(fft_input)
        self.power = fft*fft
        self.power = numpy.abs(self.power) # combine with line above?
        if plot_it:
            if title is None:
                title = 'Power spectrum'
            self._plot_freq(self.power[:blocksize/2], blocksize,
                            title=title, xlabel='Hz',
                            ylabel='power (no units)')

    def autocorrelation(self, num_points, title=None):
        """Compute and plot autocorrelation of sound file.

        Uses result from power_spectrum()
        """
        # TODO: verify alg
        if title is None:
            title = 'Autocorrelation'
        autoc = numpy.abs(numpy.fft.ifft(self.power))
        self._plot_time(autoc, offset_i=0, num_points=num_points,
                        title=title, ylabel='autocorrelation')

    def cepstrum(self, num_points, plot_it=True, title=None):
        """Compute and plot cepstrum of sound file, return pitch and goodness-of-pitch.

        Uses result from power_spectrum()
        """
        # TODO: verify alg
        cepstrum_ = numpy.abs(numpy.fft.ifft(numpy.log(self.power)))

        # compute pitch, goodness_of_pitch
        n_skip = self._n_cepstrum_points_to_skip_for_pitch
        indx_max = n_skip + cepstrum_[n_skip:num_points].argmax()
        goodness_of_pitch = cepstrum_[indx_max]
        pitch = self.sample_rate/indx_max

        if plot_it:
            if title is None:
                title = 'Cepstrum'
            cepstrum_[0] = 0 # clear the zero-shift point so it doesn't affect scale
            self._plot_time(cepstrum_, offset_i=0, num_points=num_points,
                            title=title, ylabel='cepstrum')
            plt.annotate(xy=(self.sample_times[indx_max], goodness_of_pitch),
                         s='{0:.0f}Hz'.format(pitch), color='b')

        return goodness_of_pitch, pitch

    def goodness_of_pitch(self, blocksize=1024, overlap=.50, threshold=.25,
                          plot_it=True, title=None):
        """Compute and optionally plot pitch, goodness-of-pitch, and entropy
        of sound file.
        """
        # TODO: verify alg
        dur_of_N = self.sample_times[blocksize] # how much time N represents
        end_time = self.sample_times[self.n_wav_samps-1]
        inc_t = dur_of_N - overlap*dur_of_N

        # Because we may be using overlap, the number of measurements
        # is not obvious. Since the actual number is small, just use lists.
        time = []
        goodness_of_pitch = []
        entropy = []
        pitch = []

        offset = 0.0
        while offset+dur_of_N < end_time:
            self.power_spectrum(offset_time=offset, blocksize=blocksize,
                                window_it=True, plot_it=False)
            good, p = self.cepstrum(blocksize/2, plot_it=False)
            time.append(offset+0.5*dur_of_N)
            goodness_of_pitch.append(good)
            pitch.append(p)

            # entropy
            am = self.power.mean()
            gm = numpy.exp( numpy.log(self.power).mean() )
            entropy.append(gm/am)

            offset += inc_t

        stacks = []
        if threshold:
            mx_good = max(goodness_of_pitch)
            mn_good = min(goodness_of_pitch)
            goodness_threshold = threshold*(mx_good-mn_good) + mn_good
            # identify periods with goodness-of-pitch above goodness_threshold
            # these periods are stacked harmonics (stacks for short)
            stack_pitches = [] # contains pitch values for each stack (temp)
            for i, good in enumerate(goodness_of_pitch):
                if good > goodness_threshold:
                    if stack_pitches:
                        # still in stacked harmonic
                        stack_pitches.append(pitch[i])
                    else:
                        # start of stacked harmonic
                        start_t = time[i]
                        stack_pitches.append(pitch[i])
                else:
                    # below threshold
                    if stack_pitches:
                        # close the stack
                        dur = time[i] - start_t
                        stacks.append((start_t, dur, stack_pitches))
                        stack_pitches = []
                    # else - below thr, no open stack - nothing to do
            # what if stack still open (end above thrsh)? close it
            if stack_pitches:
                dur = time[-1] - start_t
                stacks.append((start_t, dur, stack_pitches))
                stack_pitches = []

        if plot_it:
            fig, ax1 = plt.subplots(figsize=(10.0, 4.0), dpi=80)
            ax1.plot(time, pitch, 'b.')
            ax1.set_ylabel('pitch', color='b')
            for tlab in ax1.get_yticklabels():
                tlab.set_color('b')

            plt.grid(True)

            ax2 = ax1.twinx()
            ax2.plot(time, goodness_of_pitch, 'r.-')
            if threshold:
                ax2.plot([time[0], time[-1]],
                         [goodness_threshold, goodness_threshold],
                         'm--')

            # plot entropy, scale to fit with goodness-of-pitch so easy to see
            mx_good = max(goodness_of_pitch)
            mx_entropy = max(entropy)
            entropy_plot_array = numpy.array(entropy) * mx_good/mx_entropy
            ax2.plot(time, entropy_plot_array, 'g.-')

            ax2.set_xlabel('Seconds')
            ax2.set_ylabel('goodness', color='r')
            for tlab in ax2.get_yticklabels():
                tlab.set_color('r')
            ax2.set_xlim([0.0, end_time*1.01])

            if title is None:
                title = 'Goodness of pitch (entropy in green)'
            plt.title(title)

        return stacks

if __name__ == '__main__':
    # Example of how to use SignalLab.
    #
    # This can be run from IDLE (F5), but Python sometimes crashes. Would
    # running from command line or within wxPython stop that?
    #
    # path = 'C:\\Users\\Brandon\\Documents\\perkel\\motifs - clean 11-29-15\\zf3017_42337.39741472_11_29_11_2_21_motif_1.wav'
    path = 'zf3017_42337.39741472_11_29_11_2_21_motif_1.wav'
    signal_data = SignalLab(path)
    signal_data.plot_time(offset_time=0.0)
    N = 1024
    offset = 0.57 # nice harmonic stack at this location
    signal_data.plot_time(offset_time=offset, num_points=N)
    signal_data.power_spectrum(offset_time=offset, blocksize=N)
    signal_data.autocorrelation(N/2)   # good?
    signal_data.cepstrum(N/2)          # bad? no looks good!
    stacks = signal_data.goodness_of_pitch()
    for stack in stacks:
        print('stack start time: {0:.3f} dur: {1:.3f} pitches: {2}'.format(
            stack[0], stack[1], stack[2]))
    signal_data.plot_spectrogram(max_freq=10e3)
    plt.show() # shows plots and waits for user to close them all

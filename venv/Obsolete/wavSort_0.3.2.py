#!/usr/bin/env python
"""
Stripping down 8 August 2020 for manual song sorting
Version 0.3.2. Adding syllable segmentation.
Added FF analysis
Added status bar support
Added dialog for setting start and stop times for FF analysis
Added high-pass filter

This program allows sorting of WAV files into separate target directories based on manual or automated criteria
Both X and Y axes allow "auto" or "manual" settings. For Y, auto
mode sets the scaling of the graph to see all the data points.
For X, auto mode makes the graph "follow" the data. Set it X min
to manual 0 to always see the whole data from the beginning.
Note: press Enter in the 'manual' text box to make a new value 
affect the plot.
Eli Bendersky (eliben@gmail.com)
License: this code is in the public domain
Last modified: 31.07.2008
"""
try:
    import os
    import shutil
    import sys
    import wx
except ImportError:
    print('Software configuration error:.')  # % str(target)
    print('Please email perkel@uw.edu for assistance.')  # % "perkel@uw.edu"
    input('Press enter to close this window.')
    sys.exit(1)

# The recommended way to use wx with mpl is with the WXAgg
# backend. 
try:
    import matplotlib

    matplotlib.use('WXAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_wxagg import \
        FigureCanvasWxAgg as FigCanvas
    from matplotlib.ticker import AutoMinorLocator
    import numpy as np
    import pylab
    import wave
    import matplotlib
    from scipy.signal import butter, lfilter
except ImportError:
    print('Software configuration error. Some package is missing.')
    print('Please email perkel@uw.edu for assistance.')
    input('Press enter to close this window.')
    sys.exit(1)

import signal_lab

# Global Parameters
NFFT = 256  # the length of the windowing segments
noverlap = 200
debug = False


def move(src_path, subfolder):
    # move a file, specified by src_path, into subfolder
    folder = os.path.dirname(src_path)
    dest_folder = os.path.join(folder, subfolder)  # dest folder
    try:
        os.makedirs(dest_folder)
    except OSError:
        if not os.path.isdir(dest_folder):
            raise
    print("moving %s to %s" % (src_path, dest_folder))
    shutil.move(src_path, dest_folder)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class GraphFrame(wx.Frame):
    # The main frame of the application

    def __init__(self, parent, title):
        # width appears to be defined by control panel
        wx.Frame.__init__(self, parent, title=title, size=(1000, 600))  # , size=wx.Size(1000, 600))
        self.panel = wx.Panel(self)
        self.statusbar = self.CreateStatusBar()  # A Statusbar in the bottom of the window

        # Setting up the menu.
        filemenu = wx.Menu()

        # wx.ID_ABOUT and wx.ID_EXIT are standard IDs provided by wxWidgets.
        filemenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        filemenu.AppendSeparator()
        filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")  # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        # self.SetWindowStyle(wx.STAY_ON_TOP)
        self.Show(True)

        self.paused = False
        self.create_menu()
        self.create_main_panel()
        self.thisFile = 0
        self.currFilePath = ''
        self.fileList = []
        self.nFiles = 0
        self.chunkDur = 0.002  # chunk duration for segmentation (in s)
        self.fs = 0.0  # sampling rate in Hz
        self.dt = 0.0  # reciprocal of fs
        self.data = 0  # waveform data from wav file
        self.times = None
        self.spec = []
        self.currFolder = os.getcwd()

    def open_file(self, path, batch=False):
        self.signal_data = signal_lab.SignalLab(path)

        snd = wave.open(path, 'r')  # much more reliable
        nframes = snd.getnframes()
        params = snd.getparams()  # tuple (nchannels, sampwidth, framerate, nframes, comptype, compname)
        self.fs = int(params[2])  # sample rate in Hz int
        stream = snd.readframes(nframes)
        sampwidth = snd.getsampwidth()
        if sampwidth != 2:
            self.ErrorDlg('wave file %s has %d byte samples. I need 2-byte samples.' % (path, snd.getsampwidth()))
            return
        self.data = np.frombuffer(stream, dtype=np.int16)
        snd.close()
        # should check for successful read here
        self.currFilePath = path
        if debug:
            print("Read file with length ", len(self.data))
        self.dt = 1.0 / self.fs  # dt is the delta time between sampling rate
        self.times = np.arange(0, len(self.data)) * self.dt

        # low-pass filter
        order = 6
        lp_cutoff = 1e3 * float(self.low_pass_f.GetValue())
        if lp_cutoff >= self.fs / 2.0:
            lp_cutoff = 0.0
        if lp_cutoff:
            self.data = butter_lowpass_filter(self.data, lp_cutoff, self.fs, order)

        # high-pass filter to reduce noise from sound boxes
        order = 6
        hp_cutoff = float(self.high_pass_f.GetValue())
        if hp_cutoff:
            self.data = butter_highpass_filter(self.data, hp_cutoff, self.fs, order)

        # Plot waveform
        self.plot()

        self.ax2.clear()

        if lp_cutoff:
            maxfreq = lp_cutoff
        else:
            maxfreq = None
        xextent = (0, self.times[
            -1])  # this seems to be the convention - range from zero to time of last sample, even though it's probably not used
        Pxx, freqs, bins, im = signal_lab.specgram_freq_limit(self.data, NFFT=NFFT,
                                                              Fs=self.fs, noverlap=noverlap, maxfreq=maxfreq,
                                                              xextent=xextent, ax=self.ax2)

        self.spec = [Pxx, freqs, bins]
        self.ax2.set_xlim([0.0, self.times[-1] * 1.01])
        self.canvas.draw()

        self.updateStatusBar()
        self.syllable_times = []

    def create_menu(self):
        self.menubar = wx.MenuBar()

        menu_file = wx.Menu()
        m_About = menu_file.Append(wx.ID_ABOUT, "&About", " Information about this program")
        self.Bind(wx.EVT_MENU, self.OnAbout, m_About)
        m_open = menu_file.Append(-1, "&Open wav file\tCtrl-O", "Open wav file")
        self.Bind(wx.EVT_MENU, self.on_wav_open, m_open)
        menu_file.AppendSeparator()
        m_save = menu_file.Append(-1, "&Save plot\tCtrl-S", "Save plot to file")
        self.Bind(wx.EVT_MENU, self.on_save_plot, m_save)
        menu_file.AppendSeparator()
        m_exit = menu_file.Append(-1, "E&xit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)

        self.menubar.Append(menu_file, "&File")

        menu_analysis = wx.Menu()
        m_Browse = menu_analysis.Append(-1, "&Browse files\tCtrl-B", "Browse spectrograms of wave files")
        self.Bind(wx.EVT_MENU, self.on_browse, m_Browse)
        self.menubar.Append(menu_analysis, "&Analysis")

        self.SetMenuBar(self.menubar)

    def on_save_plot(self, event):
        file_choices = "PNG (*.png)|*.png"

        dlg = wx.FileDialog(
            self,
            message="Save plot as...",
            defaultDir=os.getcwd(),
            defaultFile="plot.png",
            wildcard=file_choices,
            style=wx.SAVE)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message("Saved to %s" % path)
            self.updateStatusBar()

    def create_main_panel(self):

        self.init_plot()
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        nSizeNumBox = 45  # n pix across for number input text field

        self.open_button = wx.Button(self.panel, -1, "Open", style=wx.ALIGN_BOTTOM)
        self.Bind(wx.EVT_BUTTON, self.on_wav_open, self.open_button)
        self.browse_button = wx.Button(self.panel, -1, "Browse", style=wx.ALIGN_BOTTOM)
        self.Bind(wx.EVT_BUTTON, self.on_browse, self.browse_button)
        self.prev_button = wx.Button(self.panel, -1, "<<Previous", style=wx.ALIGN_BOTTOM)
        self.Bind(wx.EVT_BUTTON, self.on_prev_file, self.prev_button)
        self.next_button = wx.Button(self.panel, -1, "Next>>", style=wx.ALIGN_BOTTOM)
        self.Bind(wx.EVT_BUTTON, self.on_next_file, self.next_button)
        self.sort_button = wx.Button(self.panel, -1, "Sort", style=wx.ALIGN_BOTTOM)
        self.Bind(wx.EVT_BUTTON, self.on_sort, self.sort_button)
        self.good_button = wx.Button(self.panel, -1, "Good", style=wx.ALIGN_BOTTOM)
        self.Bind(wx.EVT_BUTTON, self.on_good_file, self.good_button)
        self.bad_button = wx.Button(self.panel, -1, "Bad", style=wx.ALIGN_BOTTOM)
        self.Bind(wx.EVT_BUTTON, self.on_bad_file, self.bad_button)
        self.quit_button = wx.Button(self.panel, -1, "Quit", style=wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_BUTTON, self.on_quit, self.quit_button)

        self.file_num = wx.TextCtrl(self.panel, -1, '', size=(nSizeNumBox, -1))
        self.num_files = wx.TextCtrl(self.panel, -1, '', size=(nSizeNumBox, -1))
        self.low_pass_f = wx.TextCtrl(self.panel, -1, '10.0', size=(nSizeNumBox, -1))
        self.high_pass_f = wx.TextCtrl(self.panel, -1, '800', size=(nSizeNumBox, -1))

        self.panel.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
        self.panel.SetFocus()

        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox3 = wx.BoxSizer(wx.HORIZONTAL)  # holds controls for file_num, junp_button, num_files
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.hbox3, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.hbox1.Add(self.open_button, 0, flag=wx.ALIGN_LEFT | wx.BOTTOM)
        self.hbox1.Add(self.browse_button, 0, flag=wx.ALIGN_LEFT | wx.BOTTOM)
        self.hbox1.Add(self.prev_button, 0)
        self.hbox1.Add(self.next_button, 0)
        self.hbox1.Add(self.sort_button, 0)
        self.hbox1.Add(self.good_button, 0)
        self.hbox1.Add(self.bad_button, 0)
        self.hbox1.Add(self.quit_button, 0)

        self.hbox3.Add(wx.StaticText(self.panel, -1, 'File num (1st is 0):'), 0)
        self.hbox3.Add(self.file_num, 0)
        self.hbox3.Add(wx.StaticText(self.panel, -1, '   Num files:'), 0)
        self.hbox3.Add(self.num_files, 0)
        self.hbox3.Add(wx.StaticText(self.panel, -1, 'Hi Freq (kHz):'), 0)
        self.hbox3.Add(self.low_pass_f, 0)
        self.hbox3.Add(wx.StaticText(self.panel, -1, 'Hi pass filt (Hz):'), 0)
        self.hbox3.Add(self.high_pass_f, 0)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)

    def init_plot(self):
        self.dpi = 100
        self.fig = Figure((6.0, 6.0), dpi=self.dpi)

        self.axes = self.fig.add_subplot(2, 1, 1)
        # self.axes.set_axis_bgcolor('black')
        self.axes.set_title('WAV data', size=12)
        self.axes.xaxis.set_minor_locator(AutoMinorLocator())
        self.axes.tick_params(which='minor', axis='x', color='w')

        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title('spectrogram', size=12)
        self.ax2.xaxis.set_minor_locator(AutoMinorLocator())

        pylab.setp(self.axes.get_xticklabels(), fontsize=12)
        pylab.setp(self.axes.get_yticklabels(), fontsize=12)

    def on_wav_open(self, event):  # open file
        file_choices = "WAV (*.wav)|*.wav"

        dlg = wx.FileDialog(
            self,
            message="Open wav file...",
            defaultDir=self.currFolder,
            defaultFile="*.wav",
            wildcard=file_choices)
        #            style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.currFilePath = dlg.GetPath()
            self.currFolder = os.path.dirname(self.currFilePath)
            self.fileList = []
            for file in os.listdir(self.currFolder):
                if file.endswith(".wav"):
                    self.fileList.append(os.path.join(self.currFolder, file))

            self.fileList.sort()
            nf = len(self.fileList)
            self.num_files.SetValue(str(nf))

            self.open_file(self.currFilePath)
            self.nFiles = nf  # this refers to number you can browse thru
            self.num_files.SetValue(str(nf))  # this refers to number you can browse thru

            # self.file_num.SetValue(str(self.thisFile))

            # print "self.nFiles = ", self.nFiles
            # print "fileList = ", self.fileList
            # print "currFilePath = ", self.currFilePath
            self.thisFile = self.fileList.index(self.currFilePath)
            self.file_num.SetValue(str(self.thisFile))

    def on_browse(self, event):
        self.nFiles = self.getFileList()
        # print "on_browse. nf = ", self.nFiles
        if self.nFiles > 0:
            # open first file
            self.thisFile = 0
            self.open_file(self.fileList[self.thisFile])
            self.file_num.SetValue(str(self.thisFile))
        else:
            print("no files found")
            return
        self.updateStatusBar()

    def plot(self):
        self.axes.clear()

        self.time_hist_min = self.data.min()
        self.time_hist_max = self.data.max()

        self.axes.set_autoscale_on(False)
        self.axes.set_xlim([0.0, self.times[-1] * 1.01])
        self.axes.set_ylim([self.time_hist_min * 1.1, self.time_hist_max])

        self.axes.plot(self.times, self.data, 'r')

    def getFileList(self):
        # load directory list

        self.fileList = []
        dlg = wx.DirDialog(
            self, message="Choose a folder", defaultPath=self.currFolder,
        )
        if dlg.ShowModal() != wx.ID_OK:
            return 0  # as if zero files
        dirname = dlg.GetPath()
        self.currFolder = dirname

        for file in os.listdir(dirname):  # arbitrary order
            if file.endswith(".wav"):
                self.fileList.append(os.path.join(dirname, file))

        self.fileList.sort()
        nf = len(self.fileList)
        self.num_files.SetValue(str(nf))
        # print "found ", nf, " wav files"
        return nf

    def on_next_file(self, event):
        # Need to handle case when we just sorted a file and nothing is left in the directory
        if self.nFiles < 1:
            self.WarnDlg('You need to click "Browse" before you can click "Next>>"')
            return
        self.thisFile += 1
        if self.thisFile >= self.nFiles:
            self.thisFile = 0
            print('\a')  # beep
        self.open_file(self.fileList[self.thisFile])
        self.file_num.SetValue(str(self.thisFile))

    def on_prev_file(self, event):
        if self.nFiles < 1:
            self.WarnDlg('You need to click "Browse" before you can click "<<Previous"')
            return
        if self.thisFile == 0:
            self.thisFile = self.nFiles - 1
            print('\a')  # beep
        else:
            self.thisFile -= 1
        self.open_file(self.fileList[self.thisFile])
        self.file_num.SetValue(str(self.thisFile))

    def updateStatusBar(self):
        # textString = "Directory: " + os.getcwd() + "; " + "File: " + self.currFilePath
        textString = "File: " + self.currFilePath
        self.statusbar.SetStatusText(textString)

    def WarnDlg(self, s):
        dlg = wx.MessageDialog(self, s, 'Warning', wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def on_sort(self, event):
        self.nFiles = self.getFileList()
        if (debug):
            print("in sort. nf = ", self.nFiles)
            print("file list is ", self.fileList)
        for dataFile in self.fileList:
            print("datafile = " + dataFile)
            # Pxx is the segments x freqs array of instantaneous power, freqs is
            # the frequency vector, bins are the centers of the time bins in which
            # the power is computed, and im is the matplotlib.image.AxesImage
            # instance

            self.open_file(dataFile)
            # time.sleep(0.5)

            # dialog for include or exclude
            resp = self.includeOrExcludeDlg()
            print(wx.ID_OK, resp)

            # What do we do with the answer?
            if resp == wx.ID_YES:
                # Move this wav file there
                print("moving to exclude")
                move(dataFile, 'exclude')
            elif resp == wx.ID_NO:
                print("moving to include")
                move(dataFile, 'include')
            else:
                break

            self.axes.clear()
            self.canvas.draw()

            self.updateStatusBar()
        # because some files may have been moved, clear file list
        self.fileList = []
        self.nFiles = 0
        self.thisFile = 0
        self.file_num.SetValue('')
        self.num_files.SetValue('')

    def on_good_file(self, event):
        move(self.currFilePath, 'include')
        self.nFiles -= 1
        self.on_next_file(event)

    def on_bad_file(self, event):
        move(self.currFilePath, 'exclude')
        self.nFiles -= 1
        self.on_next_file(event)

    def on_exit(self, event):
        self.Destroy()

    def on_quit(self, event):
        self.Destroy()

    def onKeyPress(self, event):
        keycode = event.GetKeyCode()
        print(keycode)
        # if keycode == ord('A'):
        #    print "you pressed the spacebar!"
        #    sound_file = "notation1.wav"
        #    sound=wx.Sound(sound_file)
        #    print(sound_file)
        #    sound.Play(wx.SOUND_ASYNC)
        event.Skip()

    def OnAbout(self, e):
        # Create a message dialog box
        dlg = wx.MessageDialog(self,
                               " A simple WAV file spectrogram viewer\nDavid J. Perkel\nUniversity of Washington\nperkel@uw.edu",
                               "About wavBrowse", wx.OK)
        dlg.ShowModal()  # Shows it
        dlg.Destroy()  # finally destroy it when finished.


app = wx.App(False)
frame = GraphFrame(None, 'WavSort')
frame.SetFocus()
app.MainLoop()
del app

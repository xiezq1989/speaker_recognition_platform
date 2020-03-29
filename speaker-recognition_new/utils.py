from scipy.io import wavfile

def read_wav(fname):
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        print("convert stereo to mono")
        signal = signal[:,0]
    return fs, signal

def write_wav(fname, fs, signal):
    wavfile.write(fname, fs, signal)

def time_str(seconds):
    minutes = int(seconds / 60)
    sec = int(seconds % 60)
    return "{:02d}:{:02d}".format(minutes, sec)

def monophonic(signal):
    if signal.ndim > 1:
        signal = signal[:,0]
    return signal

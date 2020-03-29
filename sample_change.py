from ffmpy import FFmpeg
import os,sys

def sampling_tran(audio_file):
    new_audio_file=audio_file.replace('.wav','-32k.wav')
    ff = FFmpeg(
        inputs={audio_file: '-y'},
        outputs={new_audio_file: '-ar 32000'}
    )
    #print(ff.cmd)
    ff.run()
    return new_audio_file

def trans(init_path):
    for filename in os.listdir(init_path):
        whole_path=os.path.join(init_path, filename)
        if os.path.isdir(whole_path):
            trans(whole_path)
        elif os.path.isfile(whole_path):
            try:
                print(whole_path)
                sample_hz = os.popen('file %s' % whole_path).read().split(',')[4].split(' ')[2]
                sampling_rate = int(sample_hz)
                if sampling_rate != 16000:
                    sampling_tran(whole_path)
                    os.remove(whole_path)
            except Exception as e :
                print ("failed of %s "%(e))

path=sys.argv[1]
trans(path)

import os,sys


def trans(init_path):
    for dirs in os.listdir(init_path):
        whole_path=os.path.join(init_path, dirs)
        if os.path.isdir(whole_path):
            for parent, dirnames, filenames in os.walk(whole_path):
                for filename in filenames:
                    os.rename(os.path.join(parent, filename), os.path.join(parent,
                                                                           filename.replace(' ', '').replace('_',
                                                                                                             '').replace(
                                                                               '(', '_').replace(')', '')))

path=sys.argv[1]
trans(path)

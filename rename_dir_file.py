import os,sys


path=sys.argv[1]
for parent, dirnames, filenames in os.walk(path):
    for filename in filenames:
        os.rename(os.path.join(parent, filename), os.path.join(parent, filename.replace(' ', '').replace('_', '').replace('(', '_').replace(')', '')))


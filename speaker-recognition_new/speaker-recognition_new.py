#!/opt/user/anaconda3/bin/python
import os
import sys
import operator
import itertools
import glob
import argparse
from utils import read_wav
from interface import ModelInterface
from features import get_feature

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.
Examples:
    Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
    ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out
    Predict (predict the speaker of all wav files):
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll" or "predict"',
                       required=True)

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                       required=True)

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       required=True)

    parser.add_argument('-n', '--group_person_num',
                       help='The numbers of person of every group',
                       required=False)
    ret = parser.parse_args()
    return ret

def task_enroll(input_dirs, output_model,group_person_num):
    #m = ModelInterface()
    # 把输入的多个目录字符串分离为目录列表
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    # 把各个目录下的子目录列表解压出来组合成一个迭代器
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    # 生成包括所有子目录的列表
    dirs = [d for d in dirs if os.path.isdir(d)]

    # 子目录（待训练对象）总数、参数指定的每组的对象数目、组数、不够一组的余数
    objects_sum=len(dirs)
    person_num=int(group_person_num)
    group_num = objects_sum//person_num
    remainder= objects_sum % person_num
    #print(objects_sum,group_num,remainder)

    if objects_sum == 0:
        print("No valid directory found!")
        sys.exit(1)

    # 每组训练一个模型文件
    for n in range(group_num):
        start=person_num * n
        end=person_num * (n + 1)
        print('start:',start,'   end:',end)
        output_model_new=output_model+'_'+str(n)
        train_and_dump(dirs,start,end,output_model_new)

    # 最后不够一组的余数训练一个模型文件
    if remainder> 0:
        start=group_num*person_num
        end=None
        output_model = output_model + '_' + str(group_num)
        train_and_dump(dirs,start,end,output_model)

# 训练指定的目录并保存模型文件
def train_and_dump(dirs,start,end,output_model):
    m = ModelInterface()
    #print("len(dirs[start:end]):", len(dirs[start:end]))
    for d in dirs[start:end]:
        label = os.path.basename(d.rstrip('/'))
        wavs = glob.glob(d + '/*.wav')

        if len(wavs) == 0:
            print("No wav file found in %s" % (d))
            continue
        for wav in wavs:
            try:
                fs, signal = read_wav(wav)
                m.enroll(label, fs, signal)
                #print("wav %s has been enrolled" % (wav))
            except Exception as e:
                print(wav + " error %s" % (e))
    print("The group wav files has been enrolled")
    m.train()
    m.dump(output_model)
    print("%s has been dumped\t" % output_model)

def task_predict(input_files, input_model):
    # 把输入的多个模型目录字符串分离为目录列表
    input_models = [os.path.expanduser(k) for k in input_model.strip().split()]
    # 把各个目录下的模型列表解压出来组合成一个迭代器
    models = itertools.chain(*(glob.glob(m) for m in input_models))
    # 生成并加载包括所有模型文件的列表
    models = [ModelInterface.load(m) for m in models]
    # 定义统计准确率的变量
    right=0
    wrong=0
    num=0
    # 对每个预测音频文件提取特征并与每个模型匹配得到TOP结果
    for f in glob.glob(os.path.expanduser(input_files)):
        fs, signal = read_wav(f)
        print(f)
        feat = get_feature(fs, signal)
        predict_result=[]
        # 每个音频文件分别匹配每个模型组并得出分数放到总列表
        for model in models:
            #print(model)
            #m = ModelInterface.load(model)
            results = model.predict(feat)
            for result in results:
                predict_result.append(result)
        #print("predict_result:",predict_result)
        # 对预测结果按分数作高到底排序
        predict_result = sorted(predict_result, key=operator.itemgetter(1), reverse=True)
        #print("sort_predict_result:", predict_result)
        # 微信语音数据集的label格式
        label=os.path.basename(f).split('_')[0]#[6:11]
        #label=os.path.basename(f).split('(')[0]#[6:11]
        # AISHELL数据集的label格式
        #label=os.path.basename(f)[6:11]
        predict=predict_result[0][0]
        #print('Top:',predict_result[:10])
        # 统计准确率
        if label in predict:
            right+=1
            print('label:', label, '  predict:', predict, '  right')
        else:
            wrong+=1
            print('label:', label, '  predict:', predict, '  wrong')
        num+=1
    print('All:', num, '  right:', right, '  wrong:', wrong, '  acc:', right / num)


if __name__ == "__main__":
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model,args.group_person_num)
    elif task == 'predict':
        task_predict(args.input, args.model)

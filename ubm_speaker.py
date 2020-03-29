#/opt/user/anaconda3/bin/python
import os
import sys
import operator
import itertools
import glob,pickle
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
    ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out -n 300
    Predict (predict the speaker of all wav files):
    ./speaker-recognition.py -t predict -i "./*.wav" -m "model.out*"
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll" or "predict" or "mfcc"',
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
    parser.add_argument('-f', '--features_save',
                       help='The dir of features whihc want to save',
                       required=False)
    ret = parser.parse_args()
    return ret

def task_enroll(input_dirs, output_model,group_person_num=None, features_save=None):
    #m = ModelInterface()
    # 把输入的多个目录字符串分离为目录列表
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    # 把各个目录下的子目录列表解压出来组合成一个迭代器
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    # 生成包括所有子目录的列表
    dirs = [d for d in dirs if os.path.isdir(d)]

    # 子目录（待训练对象）总数
    objects_sum=len(dirs)
    if objects_sum == 0:
        print("No valid directory found!")
        sys.exit(1)

    # 如果不指定group_person_num参数则令group_person_num=objects_sum
    if group_person_num:
        person_num = int(group_person_num)
    else:
        person_num=objects_sum
        
    # 参数指定的每组的对象数目 如果小于 待训练对象总数
    if person_num<objects_sum:
        # 组数
        group_num = objects_sum // person_num
        # 不够一组的余数
        remainder= objects_sum % person_num
        print(objects_sum,group_num,remainder)

        # 每组训练一个模型文件
        for n in range(group_num):
            start = person_num * n
            end = person_num * (n + 1)
            print('start:', start, '   end:', end)
            output_model_new = output_model + '_' + str(n)
            features_save = features_save + '_' + str(n)
            train_and_dump(dirs, start, end, output_model_new, features_save)

        # 最后不够一组的余数训练一个模型文件
        if group_num>0 and remainder > 0:
            start = group_num * person_num
            end = None
            output_model_new = output_model + '_' + str(group_num)
            features_save = features_save + '_' + str(group_num)
            train_and_dump(dirs, start, end, output_model_new, features_save)
    # 如果指定数目大于等于原有数目，不分组直接训练
    else:
        train_and_dump(dirs, None, None, output_model, features_save)

# 训练指定的目录并保存模型文件和mfcc特征文件
def train_and_dump(dirs,start,end,output_model,features_save):
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
    # 如果指定了mfcc特征文件保存路径，则保存mfcc特征文件
    if features_save:
        m.mfcc_dump(features_save)
        print("The features of this group wav files has been pickle.dumped to %s" %features_save)
    m.train()
    m.dump(output_model)
    print("%s has been pickle.dumped\t" % output_model)

def task_mfcc_train(input_files, output_model):
    # 把所有mfcc特征文件统一到一个字典里面
    mfcc_dic_all = {}
    for file in glob.glob(os.path.expanduser(input_files)):
        with open(file, 'rb') as f:
            mfcc_dic = pickle.load(f)
            # 合并字典
            mfcc_dic_all = {**mfcc_dic, **mfcc_dic_all}
            #print([k for k in mfcc_dic])
    # 训练并保存模型文件
    m = ModelInterface()
    m.features=mfcc_dic_all
    m.train()
    m.dump(output_model)
    print("%s has been pickle.dumped\t" % output_model)


def task_predict(input_files, input_model):
    # 把输入的多个模型目录字符串分离为目录列表
    input_models = [os.path.expanduser(k) for k in input_model.strip().split()]
    # 把各个目录下的模型列表解压出来组合成一个迭代器
    models = itertools.chain(*(glob.glob(m) for m in input_models))
    # 生成并加载包括所有模型文件（skgmm.GMMSet object）的列表
    models = [ModelInterface.load(m) for m in models]
    if len(models) == 0:
        print("No model file found in %s" % input_model)
        sys.exit(1)
    # 定义统计准确率的变量
    right=0
    right1=0
    wrong=0
    wrong1=0
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
            results = model.predict_one(feat)
            #print(results)
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
       # label=os.path.basename(f)[6:11]
        predict=predict_result[0][0]
        predict_score=predict_result[0][1]
        # #print('Top:',predict_result[:10])
        # 统计top1准确率
        if label in predict:
            right1+=1
            print('label:', label, '  predict:', predict, '  score:', predict_score, ' top1 right')
        else:
            wrong1+=1
            print('label:', label, '  predict:', predict, '  score:', predict_score, ' top1 wrong')
        # 统计Top10准确率
        predicts = []
        predict_scores = []
        for pre in predict_result[:10]:
            predicts.append(pre[0])
            predict_scores.append(pre[1])
        if label in predicts:
            right += 1
            print('label:', label, '  predicts:', predicts, '  scores:', predict_scores, ' top10 Right')
        else:
            wrong += 1
            print('label:', label, '  predicts:', predicts, '  scores:', predict_scores, ' top10 Wrong')
        num+=1
    print('top1:', num, '  right:', right1, '  wrong:', wrong1, ' top1 acc:', right1 / num )
    print('top10:', num, '  right:', right, '  wrong:', wrong, ' top10 acc:', right / num )


if __name__ == "__main__":
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model, args.group_person_num, args.features_save)
    elif task == 'predict':
        task_predict(args.input, args.model)
    # 读取之前保存的mfcc特征进行训练
    elif task == 'mfcc':
        task_mfcc_train(args.input, args.model)

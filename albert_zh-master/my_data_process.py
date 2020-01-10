#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd


# 将原始数据文本中的中文逗号替换成英文分号，防止读csv发生错误
def text_english_to_chinese_comma(input_file='data/train.csv'):
    data_df = pd.read_csv(input_file, encoding='utf-8', header=0)
    print(data_df.head())
    data_df['title'].apply(lambda x: str(x).replace(',', '，'))
    data_df['content'].apply(lambda x: str(x).replace(',', '，'))
    data_df.to_csv('data/train3.csv')
# text_english_to_chinese_comma()
# exit()


# 打乱文件的行顺序
def shuffle_file(input_file='data/train.csv'):
    data_df = pd.read_csv(input_file, encoding='utf-8', header=0)
    print(data_df.head())
    from sklearn.utils import shuffle
    data_df = shuffle(data_df)
    data_df.to_csv('data/train_shuffle.csv', index=None, encoding='utf-8')
# shuffle_file()
# exit()


# 将bert输出文件转化为比赛所需的提交格式，is_tset区分测试与验证
def convert_to_submit(result_csv='bert_output_robert_large_2_eda_no/test_results.tsv',
                      result_file='1.csv' ,submit_file='bert_output_robert_large_2_eda_no.csv', rule=2, is_test=False):
    # result_csv='albert_large_output_checkpoints_14_2/test_results.tsv',
    import os
    print(os.path.isfile(result_csv))
    result_df = pd.read_csv(result_csv, encoding='utf-8', sep='\t', header=None)
    print(result_df.shape)
    if rule == 1:
        label_index = list(result_df.idxmax(axis=1))
    elif rule == 2:
        # 调整 0 1 类别的划分阈值
        label_index = []
        for index,row in result_df.iterrows():
            if row[0]-row[1] > 0.015:
                label_index.append(0)
            else:
                label_index.append(1)
    # print(label_index)

    test_data_df = pd.read_csv('data/test.csv', encoding='utf-8', sep=',')
    if is_test:
        test_data_df.columns = ['id', 'title', 'context']
    else:
        test_data_df.columns = ['id', 'flag', 'title', 'context']
    test_data_df['pre_label'] = label_index
    test_data_df.to_csv(result_file, encoding='utf-8', sep=',', index=None)

    submit_df = test_data_df[['id', 'pre_label']]  # pre_label需要自己手动改为flag
    submit_df.to_csv(submit_file, encoding='utf-8', sep=',', index=None)
convert_to_submit(is_test=True)
exit()


'''一下几个函数为一个整体，去除数据中的杂乱部分
预处理掉类似这种内容：
1. ';}Ie_+='';Kc_(sy_,Ie_);}}function ht_(){var Sm_=aJ_();for (sy_=0; sy_< wF_.length;aT_++){var nv_=oX_(wF_[aT_],'_');Ie_+='';}Ie_+='';Kc_(sy_,Ie_);}else{var wF_= oX_(nK_[sy_],',');var Ie_='';for (aT_=0; aT_< wF_.length;aT_++){Ie_+=Mg_(wF_[
2. 緈冨_吶庅羙 2017/02/04 18:43:53 发表在 19楼 
'''
import re

pat_case1 = r'[\'a-zA-Z_+=();}{,0-9<\s\[\]\.]{50,}'
pat_case2 = r'发表在.{0,5}(楼|板凳)'

def remove_str(instr, pat):
    patc = re.compile(pat, re.M | re.I)
    obj = patc.search(instr)
    if obj is None:
        return instr
    #     print(obj.group(0))
    pos = instr.find(obj.group(0))
    outstr = instr.replace(obj.group(0), '')
    return outstr

def process_case1(instr):
    return remove_str(instr, pat_case1)

def process_case2(instr):
    pat = re.compile(pat_case2, re.M | re.I)
    obj = pat.search(instr)
    if obj is None:
        return instr
    return instr[instr.find(obj.group(0)) + len(obj.group(0)):]

def process_str(instr):
    o_str = instr
    o_str = process_case1(o_str)
    o_str = process_case2(o_str)
    return o_str

def clean_text(id_flag_test_file, output_clean_file):
    id_flag_test_df = pd.read_csv(id_flag_test_file, encoding='utf-8', sep=',', header=0)
    id_flag_test_df['text'].apply(lambda x: process_str(x))
    id_flag_test_df.to_csv(output_clean_file, encoding='utf-8', sep=',', header=0)
# clean_text('data/train')
# exit()


def id_flag_titie_content_to_id_flag_text(id_flag_titie_content_file):
    '''将texta和textb合并'''
    id_flag_text = open("test_id_text.csv", 'w+', encoding='utf-8')
    id_flag_text.write('id,flag,text\n')
    id_flag_titie_content_df = pd.read_csv(id_flag_titie_content_file, encoding='utf-8', sep=',', header=0)
    for line in id_flag_titie_content_df.values:
        # flag = str(line[1])
        # title = str(line[2]).strip().replace(',','，')
        # content = str(line[3]).strip().replace(',','，')
        flag = '0'
        title = str(line[1]).strip().replace(',', '，')
        content = str(line[2]).strip().replace(',', '，')
        new_line = str(line[0]) + ',' + flag + ',' + title + '；' + content
        id_flag_text.write(new_line + '\n')
    id_flag_text.close()
id_flag_titie_content_to_id_flag_text('data/test.csv')
exit()


def result_merge(input_file1, input_file2, input_file3):
    '''合并3（多）个文件的标签，采用投票策略'''
    input1_df = pd.read_csv(input_file1, encoding='utf-8', sep=',', header=0)
    input2_df = pd.read_csv(input_file2, encoding='utf-8', sep=',', header=0)
    input3_df = pd.read_csv(input_file3, encoding='utf-8', sep=',', header=0)
    label_merge = []
    for i1, i2, i3 in zip(input1_df.values, input2_df.values, input3_df.values):
        num_0 = 0
        if i1[1] == 0:
            num_0 += 1
        if i2[1] == 0:
            num_0 += 1
        if i3[1] == 0:
            num_0 += 1
        if num_0 > 1:
            label_merge.append(0)
        else:
            label_merge.append(1)
    result_df = input1_df[['id']]
    result_df['flag'] = label_merge
    result_df.to_csv('result_merge.csv', index=None, encoding='utf-8', sep=',')
result_merge('pred_14_2.csv', 'pred_9.csv', 'sub.csv')
exit()


# 统计训练数据的基本信息
def data_basic_info(input_file='data/train.csv'):
    from collections import Counter

    train_data_df = pd.read_csv(input_file, encoding='utf-8', sep=',', header=0)
    print('original basic info:')
    print(train_data_df.head())
    # train_data_df.columns = ['id', 'flag', 'title', 'content']  # id,flag,title,content  ['id', 'label', 'text']
    train_data_df['context_length'] = train_data_df['content'].apply(lambda x:len(str(x)))
    train_data_df['title_length'] = train_data_df['title'].apply(lambda x:len(str(x)))
    print('basic info:')
    print('label info:')
    print(Counter(train_data_df['flag']))
    print(train_data_df.describe())
    # 单独统计验证集的信息
    # validation_data_df = pd.read_csv('data/validation_data.csv', encoding='utf-8', sep='\t')
    # validation_data_df.columns = ['id', 'label', 'text']
    # validation_data_df['length'] = validation_data_df['text'].apply(lambda x: len(x))
    # print('validation data info:')
    # print(Counter(validation_data_df['label']))
    # print(validation_data_df.describe())

    # 统计训练集和验证集的总的信息
    # data_df = train_data_df.append(validation_data_df)
    # print('total data info:')
    # print(Counter(data_df['label']))  # 统计某属性中各种类别的数量
    # print(data_df.describe())  # 统计出现次数，数值范围
    # # data_df.info()  # 展示数据结构信息 ，train为一dataframe
    # count_classes = pd.value_counts(data_df['label'], sort=True).sort_index()
    # print(count_classes)
data_basic_info()
'''
basic info:
label info
Counter({0: 2985, 1: 1008})
              flag  context_length  title_length
count  3993.000000     3993.000000   3993.000000
mean      0.252442      199.728024     24.929627
std       0.434468      520.190780     30.395124
min       0.000000        2.000000      2.000000
25%       0.000000       13.000000     10.000000
50%       0.000000       33.000000     17.000000
75%       1.000000      102.000000     27.000000
max       1.000000     7560.000000    255.000000
'''

# 将txt转换为csv，另一种方式：直接改后缀
def txt_2_csv():
    f_txt = open('data/train_data.txt', 'r', encoding='utf-8')
    f_csv = open('data/train_data.csv', 'w+', encoding='utf-8')
    for line in f_txt:
        line_list = line.split("\t")
        line_str = line_list[0] + '\t' + line_list[1] + '\t' + line_list[2]
        f_csv.write(line_str)
    f_txt.close()
    f_csv.close()


# 生成label的列表形式
def product_label_list():
    file = 'data/category.xlsx'
    label_data_df = pd.read_excel(file)
    label = label_data_df['Label Name']
    print(label)
    label_list = list(label)
    print(label_list)


#检验是否含有中文字符
def is_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


# 清洗原始数据
def clean_train_data(input_file='data/validation_data.csv', output_file='data/validation_data_clean.csv', is_test=False):
    '''
    需要清洗的样例：
    s15	Compliance with Protocol	 ④不能配合检查和治疗者；
    s16	Risk Assessment	 ?7.VAS≥3分
    1.从第一个中文字符或者英文字符开始算起
    2.如果最后一个为标签符号，则去除
    :param input_file:
    :param output_file:
    :return:
    '''
    input_data_df = pd.read_csv(input_file, encoding='utf-8', sep='\t', header=None)
    output_data = open(output_file, 'w+', encoding='utf-8')
    text_mark = ['；','：', '，', '。']
    if is_test:
        text_index = 1
    else:
        text_index = 2
    for line in input_data_df.values:
        text_start = 0
        for i, c in enumerate(line[text_index]):
            if '\u4e00' <= c <= '\u9fa5' or 'a'<=c<='z' or 'A'<=c<='Z':  #
                text_start = i
                break
        if line[text_index][-1] in text_mark:
            text_str = line[text_index][text_start:-1]
        else:
            text_str = line[text_index][text_start:]
        text_str = text_str.replace(" ",'')
        if is_test:
            line_str = line[0] + '\t' + "Disease" + '\t' + text_str
        else:
            line_str = line[0] + '\t' + line[1] + '\t' +text_str
        output_data.write(line_str + '\n')

    output_data.close()


# 比较预测值和真实值找出错误样例，test_pred_file要同时包含label和pre_label
def find_wrong(test_pred_file):
    wrong_file = open("wrong_file.csv", 'w+', encoding='utf-8')
    test_pred_df = pd.read_csv(test_pred_file, encoding='utf-8', sep='\t', header=None)
    for line in test_pred_df.values:
        if line[1] != line[-1]:
            wrong_file.write(line)
    wrong_file.close()



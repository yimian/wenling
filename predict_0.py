import keras
from keras.preprocessing import text,sequence
import jieba
import numpy as np
import xlrd
import utils

# Input parameters
max_features = 5000
max_len = 200
embedding_size = 400


l1 = []
l2 = []
pos_f_name, neg_f_name, neu_f_name = 'multi_pos.txt', 'multi_neg.txt', 'multi_neu.txt'
pos_lines = open(utils.get_corpus_path(pos_f_name)).readlines()
neu_lines = open(utils.get_corpus_path(neu_f_name)).readlines()
neg_lines = open(utils.get_corpus_path(neg_f_name)).readlines()

pos_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), pos_lines))
neu_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neu_lines))
neg_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neg_lines))


# append pos and neg and neu
for i in range(len(pos_cut_lines)):
    if i < len(neg_lines):
        l1.append(pos_cut_lines[i])
        l2.append([1, 0, 0])
        l1.append(neg_cut_lines[i])
        l2.append([0, 0, 1])
    else:
        l1.append(pos_cut_lines[i])
        l2.append([1, 0, 0])

for i in range(len(neu_cut_lines)):
    l1.append(neu_cut_lines[i])
    l2.append([0, 1, 0])

x = np.array(l1)
y = np.array(l2)

# Build vocabulary & sequences
tk = text.Tokenizer(num_words=max_features, split=" ")
tk.fit_on_texts(x)
x = tk.texts_to_sequences(x)
word_index = tk.word_index
x = sequence.pad_sequences(x, maxlen=max_len)

def input_transform(slist):
    l0 = []
    for ss in slist:
        string0 = ' '.join(jieba.cut(ss.replace('\n', '')))
        l0.append(string0)
    l0 = np.array(l0)
    input0 = tk.texts_to_sequences(l0)
    input1 = sequence.pad_sequences(input0, maxlen=max_len)
    return input1

def load_model(model_name):
    print('Loading model.')
    model = keras.models.load_model(filepath=utils.get_model_path(model_name))
    print('Loading finish.')
    return model

def predict(model, string_list):
    input1 = input_transform(string_list)
    result = model.predict(input1)
    return result

def index_to_label(index):
    if index == 0:
        return 5
    elif index == 1:
        return 3
    elif index == 2:
        return 1

if __name__ == '__main__':
    wen0 = load_model('wen0_bleeding.model')

    # wb = xlrd.open_workbook('bleeding_test.xlsx')
    # sh = wb.sheet_by_index(0)
    # text_list = []
    # bq_list = []
    # review_list = []
    # for rownum in range(1, sh.nrows):
    #     if sh.row_values(rownum)[16] != "":
    #         text_list.append(sh.row_values(rownum)[7])
    #         bq_list.append(sh.row_values(rownum)[14])
    #         review_list.append(sh.row_values(rownum)[15])
    #
    # print(len(text_list))
    # print(len(review_list))
    # predict_list = predict(wen0, text_list)
    # print(predict_list)
    # with open('predict.csv', 'w') as f:
    #     headers = ('text', 'bq_label', 'review_label', 'predict_label', 'predict_probability')
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(headers)
    #     for i in range(len(text_list)):
    #         f_csv.writerow((text_list[i], bq_list[i], review_list[i], index_to_label(np.argmax(predict_list[i])), str(predict_list[i])))







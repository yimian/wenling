import xlrd

from src import utils


def extra_split_text(pos_f, neu_f, neg_f, wb_name, sh_index, label_list):
    wb = xlrd.open_workbook(utils.get_pre_corpus_path(wb_name))
    sh = wb.sheet_by_index(sh_index)
    num_pos = 0
    num_neu = 0
    num_neg = 0
    for rownum in range(1, sh.nrows):
        text = sh.row_values(rownum)[char_num('j')]
        label = sh.row_values(rownum)[char_num('k')]
        if label == label_list[0]:
            pos_f.write(text + '\n')
            num_pos += 1
        elif label == label_list[1]:
            neu_f.write(text + '\n')
            num_neu += 1
        elif label == label_list[2]:
            neg_f.write(text + '\n')
            num_neg += 1
    print('------>' + wb_name + '<------')
    print('num_pos:', num_pos)
    print('num_neu:', num_neu)
    print('num_neg', num_neg)


def char_num(char0):
    """

    :param char0:
    :return:
    """
    dict0 = {chr(i + 97): i for i in range(26)}
    return dict0[char0]


if __name__ == '__main__':
    pos_f = open(utils.get_corpus_path('multi_pos.txt'), 'w')
    neu_f = open(utils.get_corpus_path('multi_neu.txt'), 'w')
    neg_f = open(utils.get_corpus_path('multi_neg.txt'), 'w')

    wb_name = 'human_check_oral_出血_bq(1).xlsx'
    extra_split_text(pos_f, neu_f, neg_f, wb_name, 1, [1, 0, -1])
    wb_name = 'human_check_oral_敏感_LC.xlsx'
    extra_split_text(pos_f, neu_f, neg_f, wb_name, 0, [5, 3, 1])
    wb_name = 'human_check_oral_敏感wen.xlsx'
    extra_split_text(pos_f, neu_f, neg_f, wb_name, 0, [1, 3, 2])
    wb_name = 'human_check_oral_溃疡.xlsx'
    extra_split_text(pos_f, neu_f, neg_f, wb_name, 0, [1, 0, -1])
    wb_name = 'human_check_oral_牙疼.xlsx'
    extra_split_text(pos_f, neu_f, neg_f, wb_name, 0, [1, 0, -1])
    wb_name = 'human_check_oral_红肿.xlsx'
    extra_split_text(pos_f, neu_f, neg_f, wb_name, 0, [1, 0, -1])

import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def f1_np(y_true, y_pred):
    """F1 metric.

    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    # print(true_positives)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    # print(predicted_positives)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)
    # print(possible_positives)
    """Macro_F1 metric.
    """
    macro_precision = true_positives / (predicted_positives + 1e-8)
    macro_recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-8))

    """Micro_F1 metric.
    """
    micro_precision = np.sum(true_positives) / np.sum(predicted_positives)
    micro_recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    return micro_f1, macro_f1

def top_k(test_label, test_pred, k):

    TP = 0
    recall = 0
    real_sum = 0
    metrics = 0

    sample_num_eval = len(test_pred)
    for t in range(sample_num_eval):
        predict_matrix = np.mat(test_pred[t])
        predict_score = np.mat(predict_matrix.flatten())  # 二维矩阵

        real_matrix = np.mat(test_label[t])
        real_score = np.mat(real_matrix.flatten())
        positive_num = real_score.sum()
        real_sum = real_sum + positive_num

        sort_index = np.array(np.argsort(predict_score))[0]
        # print(predict_score)
        # print(sort_index)
        predict_score[np.where(predict_score != 0)] = 0
        predict_score[:, sort_index[-k:]] = 1

        # print(predict_score)
        tp = predict_score * real_score.T

        TP = TP + tp[0, 0] / k
        recall = recall + tp[0, 0] / positive_num

    avg_precision = TP / (1 * sample_num_eval)
    mi_avg_recall = TP / real_sum
    f1 = (2 * avg_precision * mi_avg_recall) / (avg_precision + mi_avg_recall)
    # ma_avg_recall = recall / sample_num_eval

    metrics = metrics + np.array([avg_precision, mi_avg_recall, f1])

    return metrics

def get_metrics(y_true,y_pred):
    # True Positive:即y_true与y_pred中同时为1的个数
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))  # 10
    # TP = np.sum(np.multiply(y_true, y_pred)) #同样可以实现计算TP
    # False Positive:即y_true中为0但是在y_pred中被识别为1的个数
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))  # 0
    # False Negative:即y_true中为1但是在y_pred中被识别为0的个数
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  # 6
    # True Negative:即y_true与y_pred中同时为0的个数
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  # 34

    # 根据上面得到的值计算A、P、R、F1
    A = (TP + TN) / (TP + FP + FN + TN)  # y_pred与y_ture中同时为1或0
    P = TP / (TP + FP)  # y_pred中为1的元素同时在y_true中也为1
    R = TP / (TP + FN)  # y_true中为1的元素同时在y_pred中也为1
    F1 = 2 * P * R / (P + R)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="samples")
    pre = precision_score(y_true, y_pred, average="samples")
    f1 = f1_score(y_true, y_pred, average="samples")
    # print("sklearn acc, pre, recall, f1:")
    # print(acc, pre, recall, f1)

    h=hamming_loss(y_true, y_pred)
    # print("汉明损失：",h)

    z=zero_one_loss(y_true, y_pred)
    # print("0-1 损失：",z)

    c=coverage_error(y_true, y_pred)-1  # 减 1原因：看第2个参考链接
    # print("覆盖误差：",c)

    r=label_ranking_loss(y_true, y_pred)
    # print("排名损失：",r)

    a=average_precision_score(y_true, y_pred)
    # print("平均精度损失：",a)

    return A, pre, R, f1, h, z, c, r, a
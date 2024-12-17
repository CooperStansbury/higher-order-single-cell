from utils import *
import torch
from models.CHESHIRE import CHESHIRE
from models.NHP import NHP
from models.HyperSAGNN import HyperSAGNN
import config
import pandas as pd
from sklearn import metrics
from scipy.io import loadmat
from os.path import exists
from tqdm import tqdm
import time
import numpy as np

# np.random.seed(0)
# torch.manual_seed(0)

args = config.parse()


def auprc(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    return metrics.auc(recall, precision)


def train(feature, y, incidence_matrix, model, optimizer):
    model.train()
    optimizer.zero_grad()
    y_pred = model(feature, incidence_matrix)
    loss = hyperlink_score_loss(y_pred, y)
    loss.backward()
    optimizer.step()


def test(feature, y, incidence_matrix, model):
    model.eval()
    with torch.no_grad():
        y_pred = torch.squeeze(model(feature, incidence_matrix))
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        label = (y_pred >= optimal_threshold).detach().numpy()
        auc = metrics.roc_auc_score(y, y_pred.detach().numpy())
        prc = auprc(y, y_pred.detach().numpy())
        acc = metrics.accuracy_score(y, label)
        f1score = metrics.f1_score(y, label)
    return auc, prc, acc, f1score


def main(mode):
    name = ['email-Enron.mat', 'contact-high-school.mat', 'congress-bills.mat', 'NDC-classes.mat', 'BIGG-iAF1260b.mat']
    # name = ['BIGG-iAF1260b.mat']
    stat_df = pd.DataFrame()
    for i in name:
        incidence_matrix_pos = loadmat('data/' + i)
        incidence_matrix_pos = torch.tensor(incidence_matrix_pos['incidenceMatrix'], dtype=torch.float)
        incidence_matrix_neg = create_neg_incidence_matrix(incidence_matrix_pos)
        # incidence_matrix_neg_2 = create_neg_incidence_matrix(incidence_matrix_pos)
        # incidence_matrix_neg_3 = create_neg_incidence_matrix(incidence_matrix_pos)
        # incidence_matrix_neg = torch.unique(torch.cat((incidence_matrix_neg_1, incidence_matrix_neg_2, incidence_matrix_neg_3), dim=1), dim=1)
        incidence_matrix_neg = torch.unique(incidence_matrix_neg, dim=1)
        incidence_matrix = torch.cat((incidence_matrix_pos, incidence_matrix_neg), dim=1)
        y = create_label(incidence_matrix_pos, incidence_matrix_neg)
        incidence_matrix_train, y_train, incidence_matrix_test, y_test = train_test_split(incidence_matrix, y, train_size=args.train_size)
        if mode == 'cheshire':
            feature = incidence_matrix_train[:, y_train == 1]
            model = CHESHIRE(input_dim=feature.shape, emb_dim=args.emb_dim, conv_dim=args.conv_dim, k=args.k, p=args.p)
        elif mode == 'nhp':
            feature = node2vec(incidence_matrix_train[:, y_train == 1], args.emb_dim)
            model = NHP(emb_dim=args.emb_dim, conv_dim=args.conv_dim)
        else:
            feature = incidence_matrix_train[:, y_train == 1]
            model = HyperSAGNN(input_dim=feature.shape, emb_dim=args.emb_dim, conv_dim=args.conv_dim, num_heads=args.num_heads)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for _ in tqdm(range(args.max_epoch)):
            train(feature, y_train, incidence_matrix_train, model, optimizer)
        auc, prc, acc, f1score = test(feature, y_test, incidence_matrix_test, model)
        model_df = pd.DataFrame(data=np.array([auc, prc, acc, f1score]).reshape(1, 4), index=[i[:-4]], columns=['AUC', 'PRC', 'ACC', 'F1'])
        stat_df = pd.concat([stat_df, model_df])
    # stat_df.to_csv('results/type1new/' + mode + '.csv')
    if exists('results/' + mode + '.csv'):
        exist_stat_df = pd.read_csv('results/' + mode + '.csv', index_col=0)
        stat_df = pd.concat([exist_stat_df, stat_df], axis=1)
        stat_df.to_csv('results/' + mode + '.csv')
    else:
        stat_df.to_csv('results/' + mode + '.csv')


if __name__ == "__main__":
    for i in range(10):
        main(mode='cheshire')



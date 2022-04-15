import numpy as np
import tensorflow as tf
import os


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import top_k_accuracy_score as top_accuracy

def compute_error(y_true, y_pred):
    top_y_true, top_y_pred = y_true[np.nonzero(y_true > 1)], y_pred[np.nonzero(y_true > 1)]
    unique_idx = tuple(np.unique((np.nonzero(y_true > 0)[0], np.nonzero(y_true > 0)[1]), axis=1))  
    
    n_districts = y_true.shape[-1]

    mse_score = mse(np.ravel(y_true), np.ravel(y_pred))
    mae_score = mae(np.ravel(y_true), np.ravel(y_pred))
    mape_score = mape(top_y_true, top_y_pred)
    topk_acc = top_accuracy(np.argmax(y_true[unique_idx]/n_districts, axis=1), y_pred[unique_idx]/n_districts, k=round(n_districts*0.20), labels=[l for l in range(n_districts)])
    return { 'MAE': mae_score, 'MSE': mse_score, 'MAPE': mape_score, 'ACC': topk_acc }


def stepwise_error(y_true, y_pred, n_steps):    
    mae_scores, mse_scores, mape_scores, topk_accs = [], [], [], np.zeros(y_true.shape[:-1], dtype=int)
    topk_recalls = np.where(topk_accs != 0, topk_accs, np.nan)
        
    n_districts = y_true.shape[-1]
    
    for t in range(n_steps):
        y_true_t, y_pred_t = y_true[:,t,:], y_pred[:,t,:]
        top_y_true, top_y_pred = y_true_t[np.nonzero(y_true_t > 1)], y_pred_t[np.nonzero(y_true_t > 1)]
    
        mse_scores.append(mse(np.ravel(y_true_t), np.ravel(y_pred_t)))
        mae_scores.append(mae(np.ravel(y_true_t), np.ravel(y_pred_t)))
        mape_scores.append(mape(top_y_true, top_y_pred))
    
    for i in range(y_true.shape[0]):
        for t in range(n_steps):

            with tf.device('/cpu:0'):
                t_true = tf.math.top_k(y_true[i,t,:], k=round(n_districts * 0.20))
                t_true = set(t_true.indices.numpy()[[di for di, val in enumerate(t_true.values.numpy()) if val > 0]])
                t_pred = set(tf.math.top_k(y_pred[i,t,:], k=round(n_districts * 0.20)).indices.numpy())
            
                if len(t_true) > 0:
                    topk_recalls[i, t] = len(t_true.intersection(t_pred)) / len(t_true)
    
    return { 'MAE': mae_scores, 'MSE': mse_scores, 'MAPE': mape_scores, 'ACC': list(np.nanmean(topk_recalls, axis=0)), 'TOP_ACC': topk_recalls }

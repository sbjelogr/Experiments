import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
# precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:,1])
# area = auc(recall, precision)

def fit_model(model, *, X_train, y_train, X_test, y_test):

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, 
        y_train, 
        eval_metric=["logloss"], 
        eval_set=eval_set, 
        verbose=0,
    early_stopping_rounds=10)
    
    
    return model


def _eval(model, X, y, thr = 0.5):
    
    proba_0 = model.predict_proba(X)[:,0]
    proba_1 = model.predict_proba(X)[:,1]
    
    auc = round(roc_auc_score(y, proba_1),4)
    
    out = {
        "auc": auc
    }
    
    return out
    

def eval_model(model, *, X_train=None, y_train=None, X_test=None, y_test=None,thr = 0.5):
    
    if X_train is not None:
        train_res = _eval( model, X_train, y_train, thr=thr)
        print("\nTrain")
        print(train_res)
    
    if X_test is not None:
        test_res = _eval( model, X_test, y_test, thr=thr)
        print("\nTest")
        print(test_res)
    

    

def plot_logloss(model, ax = None, **plot_kwargs):
    
    results = model.evals_result_

    epochs = len(results['training']['binary_logloss'])
    x_axis = range(0, epochs)
    # plot log loss
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x_axis, results['training']['binary_logloss'], label='Train', **plot_kwargs)
    ax.plot(x_axis, results['valid_1']['binary_logloss'], label='Test', **plot_kwargs)
    ax.legend()
    # make predictions for test data

def agg_buckets(df, col, q):
    df_stats = (
        df.sort_values(by=col)
        .groupby(pd.qcut(df[col], q=q, duplicates='drop'), as_index = False)
        .agg(
            tot_def = pd.NamedAgg("y","sum"),
            tot_entries = pd.NamedAgg("y","count"),
            def_rate = pd.NamedAgg("y","mean"),
            mean_proba = pd.NamedAgg("proba_default","mean"),
            mean_proba_resc = pd.NamedAgg("proba_rescaled","mean"),
            mean_proba_oversampled = pd.NamedAgg("proba_oversampled","mean"),
            mean_proba_weightedloss = pd.NamedAgg("proba_weightedloss","mean")
        )
    )
    df_stats['tot_goods'] = df_stats['tot_entries'] - df_stats['tot_def']
    
    df_stats['pc_goods'] = df_stats['tot_goods']/df_stats['tot_goods'].sum()
    df_stats['pc_def'] = df_stats['tot_def']/df_stats['tot_def'].sum()
    
    df_stats['woe'] = np.log(df_stats['pc_goods']/df_stats['pc_def'])
    df_stats['iv'] = (df_stats['pc_goods']-df_stats['pc_def'])* df_stats['woe']
    
    columns = [
        'tot_entries','tot_goods','tot_def',
        'def_rate','mean_proba','mean_proba_resc', 'mean_proba_oversampled', 'mean_proba_weightedloss', 
        'woe','iv'
    ]
#     return df_stats
    return df_stats[columns]


def robust_pow(num_base, num_pow):
    # numpy does not permit negative numbers to fractional power
    # use this to perform the power algorithmic

    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

def focal_binary(y_true, y_pred, gamma_indct=0):
    # retrieve data from dtrain matrix
    # compute the prediction with sigmoid
    # gradient
    # complex gradient with different parts
    g1 = y_pred * (1 - y_pred)
    g2 = y_true + ((-1) ** y_true) * y_pred
    g3 = y_pred + y_true - 1
    g4 = 1 - y_true - ((-1) ** y_true) * y_pred
    g5 = y_true + ((-1) ** y_true) * y_pred
    # combine the gradient
    grad = gamma_indct * g3 * robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + \
           ((-1) ** y_true) * robust_pow(g5, (gamma_indct + 1))
    # combine the gradient parts to get hessian components
    hess_1 = robust_pow(g2, gamma_indct) + \
             gamma_indct * ((-1) ** y_true) * g3 * robust_pow(g2, (gamma_indct - 1))
    hess_2 = ((-1) ** y_true) * g3 * robust_pow(g2, gamma_indct) / g4
    # get the final 2nd order derivative
    hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
            (gamma_indct + 1) * robust_pow(g5, gamma_indct)) * g1

    return grad, hess
    
def weighted_binary_cross_entropy(y_true, y_pred, imbalance_alpha =1):
    # assign the value of imbalanced alpha
#     imbalance_alpha = self.imbalance_alpha
    # retrieve data from dtrain matrix

    # gradient
    grad = -(imbalance_alpha ** y_true) * (y_true - y_pred)
    hess = (imbalance_alpha ** y_true) * y_pred * (1.0 - y_pred)

    return grad, hess
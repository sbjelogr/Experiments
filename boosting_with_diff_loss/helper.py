import xgboost
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


class ExtendedXGboost(xgboost.XGBClassifier):
    
    def __init__(self, imbalance_alpha = 1, gamma_indct=0,
                 max_depth=3, learning_rate=0.1,
                 n_estimators=100, verbosity=1,
                 objective="binary:logistic", booster='gbtree',
                 tree_method='auto', n_jobs=1, gpu_id=0, gamma=0,
                 min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                 random_state=0, missing=None, **kwargs):
        
        self.imbalance_alpha = imbalance_alpha,
        self.gamma_indct = gamma_indct
        if objective =='user:focal':
            self.obj_func = self.focal_binary
            
        elif objective == 'user:weight':
            self.obj_func = self.weighted_binary_cross_entropy
            
        else:
            self.obj_func = objective
        
        super(ExtendedXGboost, self).__init__(
            max_depth=max_depth, learning_rate=learning_rate,
            n_estimators=n_estimators, verbosity=verbosity,
            objective=self.obj_func,
            booster=booster, tree_method=tree_method,
            n_jobs=n_jobs, gpu_id=gpu_id, gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step, subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score, random_state=random_state, missing=missing,
            **kwargs)
        
    def __repr__(self):
        return "This is extended XGboost with custom loss function"
        
    def robust_pow(self, num_base, num_pow):
        # numpy does not permit negative numbers to fractional power
        # use this to perform the power algorithmic

        return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

    
    def focal_binary(self, y_true, y_pred):
        
        gamma_indct = self.gamma_indct
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
        grad = gamma_indct * g3 * self.robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + \
               ((-1) ** y_true) * self.robust_pow(g5, (gamma_indct + 1))
        # combine the gradient parts to get hessian components
        hess_1 = self.robust_pow(g2, gamma_indct) + \
                 gamma_indct * ((-1) ** y_true) * g3 * self.robust_pow(g2, (gamma_indct - 1))
        hess_2 = ((-1) ** y_true) * g3 * self.robust_pow(g2, gamma_indct) / g4
        # get the final 2nd order derivative
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
                (gamma_indct + 1) * self.robust_pow(g5, gamma_indct)) * g1

        return grad, hess
    
    def weighted_binary_cross_entropy(self, y_true, y_pred):
        # assign the value of imbalanced alpha
        imbalance_alpha = self.imbalance_alpha
        # retrieve data from dtrain matrix

        # gradient
        grad = -(imbalance_alpha ** y_true) * (y_true - y_pred)
        hess = (imbalance_alpha ** y_true) * y_pred * (1.0 - y_pred)

        return grad, hess
        
   
def plot_roc_curve(model, X,y, ax=None,
                   title='Receiver operating characteristic', label='ROC curve ', color='darkorange', **kwargs):
    """
    Plot the roc curve
    Args:
        clf: pre-trained classifier
        X: (array or pd.Dataframe) the data to score
        y: (array or pd.Dataframe or pd.Series) the real targets
        ax: matplotlib axes object
        title: (string) Title of the grapsh
        label: (string) text to label the graph
        color: (string) specify the color of the curve

    Returns:
        ax: matplotlib axes object
    """

    proba = np.round(model.predict_proba(X)[:,1],3)
    
    roc_auc = roc_auc_score(y, proba)
    fpr, tpr, thr =  roc_curve(y, proba)
    lw = 2

    label = label + str(' (area = %0.3f)' % roc_auc)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fpr, tpr, color=color,
            lw=lw, label=label, **kwargs)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="best")
    return ax

def plot_precision_recall_curve(model, X,y, ax=None,pos_label=None,
                   title='Precision Recall Curve', label='PR curve ', color='darkorange', **kwargs):
    """
    Plot the roc curve
    Args:
        clf: pre-trained classifier
        X: (array or pd.Dataframe) the data to score
        y: (array or pd.Dataframe or pd.Series) the real targets
        ax: matplotlib axes object
        title: (string) Title of the grapsh
        label: (string) text to label the graph
        color: (string) specify the color of the curve

    Returns:
        ax: matplotlib axes object
    """

    proba = np.round(model.predict_proba(X)[:,1],3)
    
    fpr, tpr, thr =  precision_recall_curve(y, proba, pos_label = pos_label)
    lw = 2

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fpr, tpr, color=color,
            lw=lw, label=label, **kwargs)
#     ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    ax.set_title(title)
    ax.legend(loc="best")
    return ax
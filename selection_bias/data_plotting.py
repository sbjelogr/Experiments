import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def concat_data(df_score_target, df_score_selection):
    
    
    df_score_target = df_score_target.rename(columns={
        col:f"{col}_target" for col in df_score_target.columns if col.startswith("score")
                                   }
                          )
    df_score_selection = df_score_selection.rename(columns={
        col:f"{col}_sel" for col in df_score_selection.columns if col.startswith("score")
                                   }
                          )

    df_score_selection = df_score_selection.rename(columns = {"y":"is_selected"})


    exp_data = pd.concat([df_score_target, df_score_selection.drop(["col_0","col_1"],axis=1)], axis = 1)
    
    return exp_data


def plot_data(in_data):
    fig, ax = plt.subplots(1,2,figsize=(20,8))
    in_data[in_data["y"]==0].plot.scatter(x="col_0", y="col_1", ax = ax[0], color = "darkgreen");
    in_data[in_data["y"]==1].plot.scatter(x="col_0", y="col_1", ax = ax[0], color = "orange");
    in_data.plot.scatter(x="col_0", y="col_1", c=in_data["score_total_target"], 
                         cmap='jet', alpha = 0.8, ax = ax[1]);
    
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,1)
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,1)


def plot_selected_vs_all_data(sel_data, all_data):
    fig, ax = plt.subplots(1,2,figsize=(20,8))
    sel_data[sel_data["y"]==0].plot.scatter(x="col_0", y="col_1", ax = ax[0], color = "green");
    sel_data[sel_data["y"]==1].plot.scatter(x="col_0", y="col_1", ax = ax[0], color = "orange");
    all_data.plot.scatter(x="col_0", y="col_1", c=all_data["score_total_target"], 
                      cmap='jet', alpha = 0.8, ax = ax[1]);


    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,1)
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,1)

def plot_selected_vs_rejected(all_data, ax = None, all_in_one=False):
    
    if ax is None:
        if all_in_one:
            fig, ax = plt.subplots(figsize=(10,8))
        else:
            fig, ax = plt.subplots(1,2,figsize=(20,8))
    if all_in_one:
        ax_0 = ax
        ax_1 = ax
    else: 
        ax[0].set_title("selected")
        ax[1].set_title("rejected")
        ax_0 = ax[0]
        ax_1 = ax[1]
    
    sel_data = all_data[all_data['is_selected']==1]
    rej_data = all_data[all_data['is_selected']==0]
    sel_data[sel_data["y"]==0].plot.scatter(x="col_0", y="col_1", ax = ax_0, color = "green");
    sel_data[sel_data["y"]==1].plot.scatter(x="col_0", y="col_1", ax = ax_0, color = "orange");
    rej_data[rej_data["y"]==0].plot.scatter(x="col_0", y="col_1", ax = ax_1, color = "blue");
    rej_data[rej_data["y"]==1].plot.scatter(x="col_0", y="col_1", ax = ax_1, color = "red");



    ax_0.set_ylim(0,1)
    ax_0.set_xlim(0,1)
    ax_1.set_ylim(0,1)
    ax_1.set_xlim(0,1)
    
def shuffle_y(y, *, nshuff= 50):
    index_1 = y[y==1].sample(nshuff).index
    index_0 = y[y==0].sample(nshuff).index
    y.loc[index_1]=0
    y.loc[index_0]=1
    
    return y

def plot_classifier(clf,df=None,frac = 1):
    xx = np.linspace(0, 1, 100)
    yy = np.linspace(0, 1, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    probas = clf.predict_proba(Xfull)
#     probas = np.log((1-probas)/(probas+0.00001))

#     cmap =  plt.cm.get_cmap("seismic")
    cmap =  plt.cm.get_cmap("jet")

    fig, ax = plt.subplots(figsize=(10,8))

    imshow_handle = plt.imshow(probas[:, 1].reshape((100, 100)),
                               extent=(0, 1, 0, 1), origin='lower', cmap = cmap, alpha = 0.6,
#                               vmin =0, vmax=1
                              )

    if df is not None:
        df[df["y"]==1].sample(frac=frac).plot.scatter(x="col_0", y="col_1", ax = ax, color = "red", marker='*',alpha =0.5)
        df[df["y"]==0].sample(frac=frac).plot.scatter(x="col_0", y="col_1", ax = ax, color = "green", marker='*',alpha =0.5)
        
    return ax

def plot_2classifier(clf1, clf2,df=None, frac = 1):
    xx = np.linspace(0, 1, 100)
    yy = np.linspace(0, 1, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    probas1 = clf1.predict_proba(Xfull)
    probas2 = clf2.predict_proba(Xfull)
#     probas = np.log((1-probas)/(probas+0.00001))

#     cmap =  plt.cm.get_cmap("seismic")
    cmap =  plt.cm.get_cmap("jet")

    fig, ax = plt.subplots(1,2,figsize=(12,8))

    imshow_handle_1 = ax[0].imshow(probas1[:, 1].reshape((100, 100)),
                               extent=(0, 1, 0, 1), origin='lower', cmap = cmap, alpha = 0.6,
#                               vmin =0, vmax=1
                              )
    imshow_handle_2 = ax[1].imshow(probas2[:, 1].reshape((100, 100)),
                               extent=(0, 1, 0, 1), origin='lower', cmap = cmap, alpha = 0.6,
#                               vmin =0, vmax=1
                              )

    if df is not None:
        df[df["y"]==1].sample(frac=frac).plot.scatter(x="col_0", y="col_1", ax = ax[0], color = "red", marker='*',alpha =0.3)
        df[df["y"]==0].sample(frac=frac).plot.scatter(x="col_0", y="col_1", ax = ax[0], color = "green", marker='*',alpha =0.3)
        df[df["y"]==1].sample(frac=frac).plot.scatter(x="col_0", y="col_1", ax = ax[1], color = "red", marker='*',alpha =0.3)
        df[df["y"]==0].sample(frac=frac).plot.scatter(x="col_0", y="col_1", ax = ax[1], color = "green", marker='*',alpha =0.3)
        
    ax[0].set_title(f"{clf1.__class__.__name__}")
    ax[1].set_title(f"{clf2.__class__.__name__}")
#     return ax

    # circle_df_rej[circle_df_rej["y"]==0].plot.scatter(x="col_0", y="col_1", ax = ax, color = "green", marker='o');
    # ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    # plt.title("Probability")
    # plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
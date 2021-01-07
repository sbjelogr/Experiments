import numpy as np
import pandas as pd
import numpy as np
import logging

from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler

def gen_samples(n_samples = 100000, n_exp = 5, n_unif = 5, n_normal = 5, min_max_scaler_range = (0,1)):
    
#     np.random.seed(42)
    logging.info(f"Generating {n_exp} exponential features with {n_samples} rows")
    X_exp = pd.DataFrame(
        np.random.exponential(200, size = (n_samples,n_exp)),
        columns = [f'col_exp_{ix}' for ix in range(n_exp)]
    )
    
    logging.info(f"Generating {n_unif} uniform features with {n_samples} rows")
    X_uni = pd.DataFrame(
        np.random.uniform(size = (n_samples,n_unif)),
        columns = [f'col_uni_{ix}' for ix in range(n_unif)]
    )
    
    logging.info(f"Generating {n_normal} normal features with {n_samples} rows")
    X_norm = pd.DataFrame(
        np.random.normal(size = (n_samples,n_normal)),
        columns = [f'col_norm_{ix}' for ix in range(n_normal)]
    )
        
    df =  pd.concat([X_exp,X_uni,X_norm], axis = 1)
    
    
    return df
    
    
def setup_logging():
    """Set up logging for SCRABBLE.

    Args:
        settings (dict): Project settings, output of `get_settings`

    Returns:
        None
    """
    format_prefix = "%(asctime)s - %(levelname)s - %(funcName)s: %(message)s"
    format_date = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(format=format_prefix, level=getattr(logging, 'INFO'), datefmt=format_date)

    # Set py4j logger to be less verbose
    py4j_logger = logging.getLogger("py4j")
    py4j_logger.setLevel(logging.INFO)

    # Set matplotlib logger to be less verbose
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.INFO)

    return None


def shift_features(df, cols_to_shift, min_max_scaler_range = (0,1)):
    
    col_to_pass_through = [col for col in df.columns if col not in cols_to_shift]
    
    scaler = MinMaxScaler(feature_range=min_max_scaler_range)
    
    out_df = None
    
    if len(cols_to_shift)==0:
        out_df = df
    elif len(col_to_pass_through)==0:
        out_df = pd.DataFrame(
            scaler.fit_transform(df),
            columns = df.columns
        )
    elif len(col_to_pass_through)>0 and len(cols_to_shift)>0:
        df_shift = pd.DataFrame(
            scaler.fit_transform(df[cols_to_shift]),
            columns = cols_to_shift
        )
        
        df_no_shift = df[col_to_pass_through]
        
        out_df =  pd.concat([df_shift,df_no_shift], axis = 1)
        out_df = out_df[df.columns]
    else: raise ValueError("Something is wrong")
        
    return out_df

def psi(leafs_train, leafs_test, col):
    train = leafs_train[col].value_counts(normalize=True).sort_index()
    train.name='train'
    test = leafs_test[col].value_counts(normalize=True).sort_index()
    test.name='test'
    d1,d2 = pd.DataFrame(train),pd.DataFrame(test)
    d = d1.join(d2, how='outer')
    
    d['woe'] = (d['train']/d['test']).apply(lambda x: np.log(x))
    d['iv'] = (d['train'] - d['test'])* d['woe']
    
    return d['iv'].sum()


# def shift_features(df, cols_to_shift, min_max_scaler_range = (0,1)):
    
#     scale_pipe = make_pipeline(ColumnSelector(cols=cols_to_shift),
#                           MinMaxScaler(feature_range=min_max_scaler_range))

#     col_to_pass_through = [col for col in df.columns if col not in cols_to_shift]
    
#     if len(col_to_pass_through)>0 and len(cols_to_shift)>0:
#         feat_union = [
#                 ('col_scaled', scale_pipe),
#                 ('col_pass', ColumnSelector(cols=col_to_pass_through))
#             ]
#     elif len(cols_to_shift)==0:
#             feat_union = [
#                 ('col_pass', ColumnSelector(cols=col_to_pass_through))
#             ]
    
#     elif len(col_to_pass_through)==0:
#             feat_union = [
#                 ('col_scaled', scale_pipe),
#             ]
#     else: raise ValueError("Something is wrong")
    
#     pipeline = Pipeline([
#         ('feats', FeatureUnion(feat_union)),
#     ])
    
#     out = pipeline.fit_transform(df)
    
#     return pd.DataFrame(out, columns = df.columns)



    
    
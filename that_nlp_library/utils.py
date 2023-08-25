# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/07_utils.ipynb.

# %% ../nbs/07_utils.ipynb 3
from __future__ import annotations
from tqdm import tqdm
import os, sys
from pathlib import Path
from functools import partial, reduce
import dill as pickle
import torch
import numpy as np
import pandas as pd
from typing import Callable, Any
from collections.abc import Iterable
import random
import warnings

# %% auto 0
__all__ = ['HiddenPrints', 'val2iterable', 'create_dir', 'check_and_get_attribute', 'callable_name', 'print_msg', 'seed_notorch',
           'seed_everything', 'save_to_pickle', 'load_pickle', 'check_input_validation', 'check_text_leaking',
           'none2emptystr', 'lambda_batch', 'lambda_map_batch', 'augmentation_helper', 'augmentation_stream_generator',
           'func_all', 'get_dset_col_names', 'sigmoid']

# %% ../nbs/07_utils.ipynb 4
class HiddenPrints:
    "To hide print command when called"
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# %% ../nbs/07_utils.ipynb 5
def val2iterable(val,lsize=1,t='list'):
    "Convert an element (nonlist value) to an iterable. Currently support list and nparray"
    if not isinstance(val, Iterable) or isinstance(val,str):
        if t=='list': val=[val for i in range(lsize)]
        elif t=='nparray': val=np.repeat(val,lsize)
        else:
            raise ValueError('Unrecognized iterable to convert to')
    return val

def create_dir(path_dir):
    "Create directory if needed"
    path_dir = Path(path_dir)
    if not path_dir.exists():
        path_dir.mkdir(parents=True)

# %% ../nbs/07_utils.ipynb 6
def check_and_get_attribute(obj,attr_name):
    if hasattr(obj,attr_name): return getattr(obj,attr_name)
    else: raise ValueError(f"Missing required argument: {attr_name}")
            
def callable_name(any_callable: Callable[..., Any]) -> str:
    "To get name of any callable"
    if hasattr(any_callable, '__name__'):
        return any_callable.__name__
    if isinstance(any_callable, partial):
        return any_callable.func.__name__

    return str(any_callable)

def print_msg(msg,dash_num=5,verbose=True):
    if verbose:
        print('-'*dash_num+' '+msg+' '+ '-'*dash_num)

def seed_notorch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
def seed_everything(seed=42):
    seed_notorch(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# %% ../nbs/07_utils.ipynb 7
def save_to_pickle(my_list,fname,parent='pickle_files'):
    if fname[-4:]!='.pkl': fname+='.pkl'
    p = Path(parent)
    p.mkdir(parents=True, exist_ok=True)
    fname = p/fname
    with open(fname, 'wb') as f:
        pickle.dump(my_list, f)
        
def load_pickle(fname,parent='pickle_files'):
    if fname[-4:]!='.pkl': fname+='.pkl'
    with open(Path(parent)/fname,'rb') as f:
        my_list = pickle.load(f)
    return my_list

# %% ../nbs/07_utils.ipynb 8
def check_input_validation(df:pd.DataFrame,verbose=True):
    verboseprint = print if verbose else lambda *a, **k: None
    print_msg('Input Validation Precheck',verbose)
    # check whether index is sorted
    correct_idxs = np.arange(df.shape[0])
    curr_idxs = df.index.values
    if not np.array_equal(correct_idxs,curr_idxs):
        verboseprint("DataFrame Index is not RangeIndex, and will be converted")
        df.index = correct_idxs

    # Do a NA check:
    na_check = df.isna().sum(axis=0)
    na_check = na_check[na_check!=0]
    if na_check.shape[0]!=0:
        verboseprint("Data contains missing values!")
        verboseprint('-----> List of columns and the number of missing values for each')
        verboseprint(na_check)

    # Do a row duplication check
    _df = df.copy().astype(str)
    dup_check = _df.value_counts(dropna=False)
    dup_check=dup_check[dup_check>1]
    if dup_check.shape[0]!=0:
        verboseprint("Data contains duplicated values!")
        verboseprint(f'-----> Number of duplications: {(dup_check.values-1).sum()} rows')
        

# %% ../nbs/07_utils.ipynb 9
def check_text_leaking(trn_txt:list,
                       test_txt:list,verbose=True):
    verboseprint = print if verbose else lambda *a, **k: None
    test_txt_leaked = {i.strip().lower() for i in trn_txt} & {j.strip().lower() for j in test_txt}
    len_leaked = len(test_txt_leaked)
    verboseprint(f'- Number of rows leaked: {len_leaked}, which is {100*len_leaked/len(trn_txt):.2f}% of training set')
    return test_txt_leaked

# %% ../nbs/07_utils.ipynb 10
def none2emptystr(x):
    if x is None: return ''
    return str(x)

# %% ../nbs/07_utils.ipynb 11
def lambda_batch(inp, # HuggingFace Dataset
                     feature, # Feature name.
                     func, # The function to apply
                     is_batched, # Whether batching is applied
                    ):
    return [func(v) for v in inp[feature]] if is_batched else func(inp[feature])

# %% ../nbs/07_utils.ipynb 12
def lambda_map_batch(inp, # HuggingFace Dataset
                     feature, # Feature name.
                     func, # The function to apply
                     is_batched, # Whether batching is applied
                     output_feature=None, # New feature output, if different from 'feature'
                     is_func_batched=False # Whether the func above only works with batch
                    ):
    results={}
    if output_feature is None: output_feature = feature
    if not is_func_batched:
        results[output_feature] = lambda_batch(inp,feature,func,is_batched)
    else:
        results[output_feature] = func(inp[feature]) if is_batched else func([inp[feature]])
    return results

# %% ../nbs/07_utils.ipynb 13
def augmentation_helper(inp,text_name,func):
    # inp[text_name] will be list
    inp[text_name]=[func(v) for v in val2iterable(inp[text_name])]
    return inp

def augmentation_stream_generator(dset,text_name,func):
    for inp in dset:
        # inp[text_name] will be a single item
        inp[text_name]=func(inp[text_name])
        yield inp

# %% ../nbs/07_utils.ipynb 14
def func_all(x, functions):
    return reduce(lambda acc, func: func(acc), functions, x)

# %% ../nbs/07_utils.ipynb 15
def get_dset_col_names(dset):
    if dset.column_names is not None: return dset.column_names
    warnings.warn("Iterable Dataset might contain multiple mapping functions; getting column names can be time and memory consuming") 
    return list(next(iter(dset)).keys())

# %% ../nbs/07_utils.ipynb 16
def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    
    Source: assignment3 of cs231n
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

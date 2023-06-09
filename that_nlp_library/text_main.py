# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_text_main.ipynb.

# %% ../nbs/00_text_main.ipynb 3
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from datasets import DatasetDict,Dataset
from pathlib import Path
from tqdm import tqdm
from .utils import *
from functools import partial

# %% auto 0
__all__ = ['tokenizer_explain', 'two_steps_tokenization_explain', 'tokenize_function', 'datasetdictize_given_idxs',
           'TextDataMain']

# %% ../nbs/00_text_main.ipynb 6
def tokenizer_explain(inp, # Input sentence
                      tokenizer, # Tokenizer (preferably from HuggingFace)
                      split_word=False # Is input `inp` split into list or not
                     ):
    "Display results from tokenizer"
    print('----- Tokenizer Explained -----')
    print('--- Input ---')
    print(inp)
    print()
    print('--- Tokenized results --- ')
    print(tokenizer(inp,is_split_into_words=split_word))
    print()
    tok = tokenizer.encode(inp,is_split_into_words=split_word)
    print('--- Results from tokenizer.convert_ids_to_tokens ---')
    print(tokenizer.convert_ids_to_tokens(tok))
    print()
    print('--- Results from tokenizer.decode --- ')
    print(tokenizer.decode(tok))
    print()

# %% ../nbs/00_text_main.ipynb 20
def two_steps_tokenization_explain(inp, # Input sentence
                                   tokenizer, # Tokenizer (preferably from HuggingFace)
                                   split_word=False, # Is input `inp` split into list or not
                                   content_tfms=[] # A list of text transformations
                                  ):
    "Display results form each content transformation, then display results from tokenizer"
    print('----- Text Transformation Explained -----')
    print('--- Raw sentence ---')
    print(inp)
    for tfm in content_tfms:
        print_msg(callable_name(tfm),3)
        inp = tfm(inp)
        print(inp)
    print()
    tokenizer_explain(inp,tokenizer,split_word)

# %% ../nbs/00_text_main.ipynb 30
def tokenize_function(examples:dict,
                      tok,
                      max_length=None, # pad to model's allowed max length (default is max_sequence_length)
                      is_split_into_words=True):
    if max_length:
        return tok(examples["text"], padding=True, max_length=max_length,truncation=True,is_split_into_words=is_split_into_words)
    return tok(examples["text"], padding="max_length", truncation=True,is_split_into_words=is_split_into_words)

# %% ../nbs/00_text_main.ipynb 31
def datasetdictize_given_idxs(kv_pairs:dict, # Dictionary; keys can be content, label, metadata. Values are list each.
                              trn_idx=None, # Training indices
                              val_idx=None, # Validation indices
                              tokenizer=None, # HuggingFace tokenizer
                              is_split_into_words=False, # Is text (content) split into list or not
                              max_length=None # pad to model's allowed max length (default is max_sequence_length)
                             ):
    "Create a HuggingFace DatasetDict with given arguments"
    if 'text' not in kv_pairs.keys():
        raise ValueError('Dictionary must have `text` (which contains texture contents) as key')
    all_dataset = Dataset.from_dict(kv_pairs)
    main_ddict = DatasetDict()
    if trn_idx is None:
        main_ddict['train'] = all_dataset
    else:
        main_ddict['train'] = all_dataset.select(trn_idx)

    if val_idx is not None:  
        main_ddict['validation'] = all_dataset.select(val_idx)
    
    print_msg("Map Tokenize Function",20)
    main_ddict_tokenized = main_ddict.map(partial(tokenize_function,
                                                  tok=tokenizer,
                                                  is_split_into_words=is_split_into_words,
                                                  max_length=max_length),batched=True)
    
    return main_ddict_tokenized

# %% ../nbs/00_text_main.ipynb 45
class TextDataMain():
    def __init__(self,
                 df: pd.DataFrame, # The main dataframe
                 main_content:str, # Name of the text column
                 metadatas=[], # Names of the metadata columns
                 label_names=None, # Names of the label (dependent variable) columns
                 class_names_predefined=None, # (Optional) List of names associated with the labels (same index order)
                 val_ratio:list|float|None=0.2, # Ratio of data for validation set. If given a list, validation set will be chosen based on indices in this list
                 split_cols:list|str=None, # Column(s) needed to do stratified shuffle split
                 content_tfms=[], # A list of text transformations
                 aug_tfms=[], # A list of text augmentations
                 process_metadatas=True, # Whether to do simmple text processing on the chosen metadatas
                 seed=None, # Random seed
                 cols_to_keep=None, # Columns to keep after all processings
                 shuffle_trn=True # Whether to shuffle the train set
                ):
        self.df = df.copy()
        self.main_content = main_content
        self.metadatas = metadatas
        self.label_names = label_names
        self.label_lists = class_names_predefined
        self.content_tfms = content_tfms
        self.aug_tfms = aug_tfms
        self.process_metadatas = process_metadatas
        self.val_ratio=val_ratio
        self.split_cols=split_cols
        self.seed = seed
        self.cols_to_keep = cols_to_keep
        self.shuffle_trn=shuffle_trn  
        self._main_called=False
        self.is_multilabel=False
        self.is_multihead=False
        check_input_validation(self.df)
        
    @classmethod
    def from_csv(cls,path,return_df=False,encoding='utf-8-sig',**kwargs):
        df = pd.read_csv(path,encoding=encoding,engine='pyarrow')
        tdm = TextDataMain(df,main_content=None) if return_df else TextDataMain(df,**kwargs)
        if return_df:
            return df
        return tdm
    
    @classmethod
    def from_pickle(cls,
                    fname, # Name of the pickle file
                    parent='pickle_files' # Parent folder
                   ):
        return load_pickle(fname,parent=parent)
    
    @classmethod
    def from_gsheet(cls,gs_id,return_df=False,**kwargs):
        pass

    
    def save_as_pickles(self,
                        fname, # Name of the pickle file
                        parent='pickle_files', # Parent folder
                        drop_data_attributes=False # Whether to drop all large-size data attributes
                       ):
        if drop_data_attributes:
            if hasattr(self, 'df'):
                del self.df
            if hasattr(self, 'main_ddict'):
                del self.main_ddict
        save_to_pickle(self,fname,parent=parent)

        
    def _check_validation_leaking(self,trn_idxs,val_idxs):
        if self.val_ratio is None:
            return trn_idxs,None
        
        df_trn = self.df.loc[trn_idxs]
        df_val = self.df.loc[val_idxs]
        
        #sanity check
        assert df_trn.shape[0]+df_val.shape[0]==self.df.shape[0],"Train + Validation != Total Data"

        
        print(f'Previous Validation Percentage: {round(100*len(val_idxs)/self.df.shape[0],3)}%')
        val_content_series = check_text_leaking(df_trn[self.main_content],df_val[self.main_content])
        val_idxs2 = val_content_series.index.values
        trn_idxs2 = self.df[~self.df.index.isin(val_idxs2)].index.values
        print(f'Current Validation Percentage: {round(100*len(val_idxs2)/self.df.shape[0],3)}%')
        if len(val_idxs2)!=len(val_idxs):
            return trn_idxs2,val_idxs2
        return trn_idxs,val_idxs
    
    def _train_test_split(self):
        print_msg('Train Test Split',20)
        rng = np.random.default_rng(self.seed)
        if self.val_ratio is None: # no train/val split
            trn_idxs = rng.permutation(self.df.shape[0])
            return trn_idxs,None
        if isinstance(self.val_ratio,list) or isinstance(self.val_ratio,np.ndarray):
            val_idxs = np.array(self.val_ratio)
            trn_idxs = np.array(set(self.df.index.values) - set(self.val_ratio))
            return trn_idxs,val_idxs
        if isinstance(self.val_ratio,float) and self.split_cols is None:
            _idxs = rng.permutation(self.df.shape[0])
            _cutoff = int(self.val_ratio*self.df.shape[0]) 
            val_idxs = _idxs[:_cutoff]
            trn_idxs = _idxs[_cutoff:]
            return trn_idxs,val_idxs
        
        self.split_cols = val2iterable(self.split_cols)
        if self.is_multilabel and self.label_names[0] in self.split_cols:
            raise ValueError('For MultiLabel classification, you cannot choose the label as your shuffle-split column')
        
        if len(self.split_cols)>0:
            _y = self.df[self.split_cols[0]]
            if len(self.split_cols)>1:
                for c in self.split_cols[1:]:
                    _y= _y.astype(str) + '_' + self.df[c].astype(str)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_ratio, 
                                         random_state=self.seed)
            trn_idxs,val_idxs = list(sss.split(self.df,_y))[0]
            return trn_idxs,val_idxs
        
        raise ValueError('No valid keyword arguments for train validation split!')

                         
    def _encode_labels(self):
        print_msg('Label Encoding')
        if self.label_names is None: 
            raise ValueError('Missing label columns!')
        self.label_names = val2iterable(self.label_names)
        if len(self.label_names)>1:
            self.is_multihead=True
        
        if self.label_lists is not None and not isinstance(self.label_lists[0],list):
            self.label_lists = [self.label_lists]
        
        if isinstance(self.df[self.label_names[0]].iloc[0],list):
            # This is multi-label. Ignore self.label_names[1:]
            self.label_names = [self.label_names[0]]
            self.is_multihead=False
            self.is_multilabel=True
            
        encoder_classes=[]
        if not self.is_multilabel:
            for idx,l in enumerate(self.label_names):
                if self.label_lists is None:
                    train_label = self.df[l].values
                    l_encoder = LabelEncoder()
                    self.df[l] = l_encoder.fit_transform(train_label)
                    encoder_classes.append(list(l_encoder.classes_))
                else:
                    l_classes = sorted(list(self.label_lists[idx]))
                    label2idx = {v:i for i,v in enumerate(l_classes)}
                    self.df[l] = self.df[l].map(label2idx).values
                    encoder_classes.append(l_classes)
        else:
            # For MultiLabel, we only save the encoder classes without transforming the label itself to one-hot (or actually, few-hot)
            if self.label_lists is None:
                l_encoder = MultiLabelBinarizer()
                _ = l_encoder.fit(self.df[self.label_names[0]])
                encoder_classes.append(list(l_encoder.classes_))
            else:
                l_classes = sorted(list(self.label_lists[0]))
                encoder_classes.append(l_classes)
                
        self.label_lists = encoder_classes
            
    def _process_metadatas(self,df,override_dict=True):
        print_msg('Metadata Simple Processing & Concatenating to Main Content')
        self.metadatas = val2iterable(self.metadatas)
            
        for s in self.metadatas:
            if self.process_metadatas:
                # just strip and lowercase
                df[s] = df[s].astype(str).str.strip().str.lower()
            # simple concatenation with '. '
            df[self.main_content] = df[s] + ' - ' + df[self.main_content]
                
        if override_dict:        
            self.metadata_dict={}
            for s in self.metadatas:
                self.metadata_dict[s]=sorted(set(df[s].values))
        return df
    
    def _simplify_df(self):
        if self.cols_to_keep is None:
            self.cols_to_keep= [self.main_content] + self.metadatas + self.label_names
        self.df = self.df[self.cols_to_keep].copy()
    
    def _do_transformation(self,df):
        print_msg('Text Transformation',20)
        for tfm in self.content_tfms:
            print_msg(callable_name(tfm))
            df[self.main_content] = [tfm(s) for s in tqdm(df[self.main_content].values)]
        return df
    
    def _do_augmentation(self,df_trn_org):
        df_trn_all = df_trn_org.copy()
        print_msg('Text Augmentation',20)
        print(f'Train data size before augmentation: {len(df_trn_all)}')
        for tfm in self.aug_tfms:
            print_msg(callable_name(tfm))
            if tfm.keywords['apply_to_all']:
                new_content,new_others = tfm(content=df_trn_all[self.main_content].values,others=df_trn_all.iloc[:,1:])
            else:
                new_content,new_others = tfm(content=df_trn_org[self.main_content].values,others=df_trn_org.iloc[:,1:])
            
            # add axis to np array in order to do concatenation
            if len(new_content.shape)==1:
                new_content = new_content[:,None]
            if len(new_others.values.shape)==1:
                new_others = new_others.values[:,None]
                
            df_tmp = pd.DataFrame(np.concatenate((new_content,new_others.values),axis=1),columns=df_trn_org.columns.values)
            df_trn_all = pd.concat((df_trn_all,df_tmp),axis=0).reset_index(drop=True)
            print(f'Train data size after THIS augmentation: {len(df_trn_all)}')       
        print(f'Train data size after ALL augmentation: {len(df_trn_all)}')
        return df_trn_all
    
    def _main_text_processing(self):
        print_msg('Start Main Text Processing',20)
        
        # Process metadatas
        self.df = self._process_metadatas(self.df)
        
        # Process labels
        self._encode_labels()
        
        # Content transformation
        self.df = self._do_transformation(self.df)
        
        # Train Test Split
        trn_idxs,val_idxs = self._train_test_split()
        self._simplify_df()
        trn_idxs,val_idxs = self._check_validation_leaking(trn_idxs,val_idxs)
        if self.val_ratio is not None:
            df_val = self.df.loc[val_idxs].reset_index(drop=True)
        
        # Augmentation
        df_trn_org = self.df.loc[trn_idxs].reset_index(drop=True)
        df_trn_all = self._do_augmentation(df_trn_org)
        df_trn_all['is_valid']=False
        
        # Shuffle train
        if self.shuffle_trn:
            df_trn_all = df_trn_all.sample(frac=1.,random_state=self.seed)
            
        # Combine augmented train and val
        if self.val_ratio is not None:
            df_val['is_valid']=True
            df_trn_all = pd.concat((df_trn_all,df_val),axis=0)
        
        self._main_called=True
        self.df = df_trn_all.reset_index(drop=True)        
    
    def set_data_collator(self,data_collator):
        self.data_collator = data_collator
        
    def tokenizer_explain_single(self,tokenizer):
        inp = self.df[~self.df['is_valid']][self.main_content].sample(1).values[0]
        tokenizer_explain(inp,tokenizer)
        
    def to_df(self): 
        "To execute all the defined processings and return a dataframe"
        if not self._main_called:
            self._main_text_processing()
        return self.df
       
    def save_train_data_after_processing(self,output_path,encoding='utf-8-sig'):
        if not self._main_called:
            print_msg('WARNING')
            print('Please process training data (using to_df or to_datasetdict)')
            return
        self.df.to_csv(Path(output_path),encoding=encoding,index=False)
    
    def to_datasetdict(self,
                       tokenizer, # Tokenizer (preferably from HuggingFace)
                       is_split_into_words=False, # Is text split into list or not
                       max_length=None, # pad to model's allowed max length (default is max_sequence_length)
                       trn_ratio=1., # Portion of training data to be converted to datasetdict. Useful for sample experiments
                       seed=42 # Random seed
                      ):
        if not self._main_called:
            self._main_text_processing()
        val_idx = self.df[self.df['is_valid']].index.values if self.val_ratio is not None else None
        trn_idx = self.df[~self.df['is_valid']].index.values
        if trn_ratio<1. and trn_ratio>0.:
            rng = np.random.default_rng(self.seed)
            _idxs = rng.permutation(len(trn_idx))
            _cutoff = int(trn_ratio*len(trn_idx)) 
            trn_idx = _idxs[:_cutoff]
            
        _label = self.df[self.label_names].values.tolist()
        if not self.is_multilabel:
            if len(self.label_names)==1:
                _label = np.array(_label).flatten().tolist() # (n,)
        else:
            # For MultiLabel, this is where the actual label transformation happens
            mlb = MultiLabelBinarizer(classes=self.label_lists[0])
            _label = self.df[self.label_names[0]].values.tolist()
            _label = mlb.fit_transform(_label).tolist() # few-hotted
        
        kv_pairs = {'text':self.df[self.main_content].tolist(),
                    'label':_label,
                   }
        for c in self.cols_to_keep:
            if c not in self.label_names+[self.main_content]: kv_pairs[c] = self.df[c].tolist()
        
        self.tokenizer = tokenizer
        self.is_split_into_words= is_split_into_words
        self.max_length = max_length
        
        ddict = datasetdictize_given_idxs(kv_pairs,trn_idx,val_idx,self.tokenizer,
                                         is_split_into_words=is_split_into_words,max_length=max_length)
        self.main_ddict = ddict
        return ddict
    
    def get_test_datasetdict_from_csv(self,path,encoding='utf-8-sig'):
        df_test = pd.read_csv(path,encoding=encoding,engine='pyarrow')
        return self.get_test_datasetdict_from_df(df_test)

    def get_test_datasetdict_from_dict(self,content):
        if len(self.metadatas)!=0 and not isinstance(content,dict):
            raise ValueError(f'There is/are metadatas in the preprocessing step. Please include a dictionary including these keys for metadatas: {self.metadatas}, and texture content: {self.main_content}')
            
        _dic = {self.main_content:[content]} if isinstance(content,str) else content
        for k in _dic.keys():
            _dic[k] = val2iterable(_dic[k])
        
        df_test = pd.DataFrame.from_dict(_dic)
        return self.get_test_datasetdict_from_df(df_test)
    
    def get_test_datasetdict_from_df(self,df_test):
        print_msg('Getting Test Set',20)
        check_input_validation(df_test)
        
        cols_to_keep = [c for c in self.cols_to_keep if c not in self.label_names]
        df_test = df_test[cols_to_keep].copy()
        
        print_msg('Start Test Set Transformation',20)
        df_test = self._process_metadatas(df_test,override_dict=False)
        df_test = self._do_transformation(df_test)
        
        if hasattr(self,'df'):
            print_msg('Test Leak Checking',20)
            _ = check_text_leaking(self.df[self.main_content],df_test[self.main_content])
        
        print_msg('Construct DatasetDict',20)
        test_text = df_test[self.main_content].values
        
        kv_pairs ={'text':test_text}
        for c in self.cols_to_keep:
            if c not in self.label_names+[self.main_content]: kv_pairs[c] = df_test[c].tolist()
        
        test_dataset = Dataset.from_dict(kv_pairs)
        test_ddict = DatasetDict()
        test_ddict['test'] = test_dataset
        test_ddict_tokenized = test_ddict.map(partial(tokenize_function,tok=self.tokenizer,
                                                      is_split_into_words=self.is_split_into_words,
                                                      max_length=self.max_length),batched=True)
        
        return test_ddict_tokenized

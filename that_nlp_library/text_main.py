# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_text_main.ipynb.

# %% ../nbs/00_text_main.ipynb 3
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from datasets import DatasetDict,Dataset,IterableDataset,load_dataset,concatenate_datasets
from pathlib import Path
from tqdm import tqdm
from .utils import *
from functools import partial
import warnings

# %% auto 0
__all__ = ['tokenizer_explain', 'two_steps_tokenization_explain', 'tokenize_function', 'concat_metadatas', 'TextDataController']

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
                                   content_tfms=[], # A list of text transformations
                                   aug_tfms=[], # A list of text augmentation 
                                  ):
    "Display results form each content transformation, then display results from tokenizer"
    print('----- Text Transformation Explained -----')
    print('--- Raw sentence ---')
    print(inp)
    print('--- Content Transformations (on both train and test) ---')
    content_tfms = val2iterable(content_tfms)
    for tfm in content_tfms:
        print_msg(callable_name(tfm),3)
        inp = tfm(inp)
        print(inp)
    print('--- Augmentations (on train only) ---')
    aug_tfms = val2iterable(aug_tfms)
    for tfm in aug_tfms:
        print_msg(callable_name(tfm),3)
        inp = tfm(inp)
        print(inp)
    print()
    tokenizer_explain(inp,tokenizer)

# %% ../nbs/00_text_main.ipynb 41
def tokenize_function(examples:dict,
                      tok,
                      text_name,
                      max_length=None,
                      is_split_into_words=False):
    if max_length is None:
        # pad to model's default max sequence length
        return tok(examples[text_name], padding="max_length", truncation=True,is_split_into_words=is_split_into_words)
    if isinstance(max_length,int) and max_length>0:
        # pad to max length of the current batch, and start truncating at max_length
        return tok(examples[text_name], padding=True, max_length=max_length,truncation=True,is_split_into_words=is_split_into_words)
    
    # no padding (still truncate at model's default max sequence length)
    return tok(examples[text_name], truncation=True,is_split_into_words=is_split_into_words)

# %% ../nbs/00_text_main.ipynb 53
def concat_metadatas(dset:dict, # HuggingFace Dataset
                     main_text, # Text feature name
                     metadatas, # Metadata (or a list of metadatas)
                     process_metas=True, # Whether apply simple metadata processing, i.e. space strip and lowercase
                     sep='.', # separator for contatenating to main_text
                     is_batched=True, # whether batching is applied
                    ):
    """
    Extract, process (optional) and concatenate metadatas to the front of text
    """
    results={main_text:dset[main_text]}
    for m in metadatas:
        m_data = dset[m]
        if process_metas:
            # just strip and lowercase
            m_data = [none2emptystr(v).strip().lower() for v in m_data] if is_batched else nan2emptystr(m_data).strip().lower()
        results[m]=m_data
        if is_batched:
            results[main_text] = [f'{m_data[i]} {sep} {results[main_text][i]}' for i in range(len(m_data))]
        else:
            results[main_text] = f'{m_data} {sep} {results[main_text]}'
    return results

# %% ../nbs/00_text_main.ipynb 57
class TextDataController():
    def __init__(self,
                 inp, # HuggingFainpce Dataset or DatasetDict
                 main_text:str, # Name of the main text column
                 label_names=None, # Names of the label (dependent variable) columns
                 class_names_predefined=None, # List of names associated with the labels (same index order)
                 filter_dict={}, # A dictionary: {feature: filtering_function_based_on_the_feature}
                 metadatas=[], # Names of the metadata columns
                 process_metas=True, # Whether to do simple text processing on the chosen metadatas
                 content_transformations=[], # A list of text transformations
                 val_ratio:list|float|None=0.2, # Ratio of data for validation set. If given a list, validation set will be chosen based on indices in this list
                 stratify_cols=[], # Column(s) needed to do stratified shuffle split
                 upsampling_list={}, # A list of tuple. Each tuple: (feature,upsampling_function_based_on_the_feature)
                 content_augmentations=[], # A list of text augmentations
                 seed=None, # Random seed
                 is_batched=True, # Whether to perform operations in batch
                 batch_size=1000, # Batch size, for when is_batched is True
                 num_proc=4, # Number of process for multiprocessing
                 cols_to_keep=None, # Columns to keep after all processings
                 buffer_size=10000, # For shuffling data
                 num_shards=64, # Number of shards. Stream datasets can be made out of multiple shards
                 convert_training_to_iterable=True, # Whether to convert training Dataset to IterableDataset
                 verbose=True, # Whether to print processing information
                ):
            
        self.main_text = main_text
        self.metadatas = val2iterable(metadatas)
        self.process_metas = process_metas
        self.label_names = val2iterable(label_names) if label_names is not None else None
        self.label_lists = class_names_predefined
        self.filter_dict = filter_dict
        self.content_tfms = val2iterable(content_transformations)
        self.upsampling_list = upsampling_list
        self.aug_tfms = val2iterable(content_augmentations)
        self.val_ratio = val_ratio
        self.stratify_cols = val2iterable(stratify_cols)
        self.seed = seed
        self.is_batched = is_batched
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.is_streamed = False
        self.cols_to_keep = cols_to_keep
        self.buffer_size = buffer_size
        self.num_shards = num_shards
        self.ddict_rest = DatasetDict()
        self.convert_training_to_iterable = convert_training_to_iterable
        self.verbose = verbose
        self.verboseprint = print if verbose else lambda *a, **k: None
        
        if hasattr(inp,'keys'):
            if 'train' in inp.keys(): # is datasetdict
                self.ddict_rest = inp
                self.dset = self.ddict_rest.pop('train')
            else:
                raise ValueError('The given DatasetDict has no "train" split')
        else: # is dataset
            self.dset = inp
        if isinstance(self.dset,IterableDataset):
            self.is_streamed=True
        self.all_cols = get_dset_col_names(self.dset)
        if self.is_streamed and self.label_names is not None and self.label_lists is None:
            raise ValueError('All class labels must be provided when streaming')
        
        if self.is_streamed and len(self.upsampling_list):
            warnings.warn("Upsampling requires dataset concatenation, which can be extremely slow (x2) for streamed dataset")
            
        self._processed_call=False
        
        self._determine_multihead_multilabel()
        
            
    @classmethod
    def from_csv(cls,file_path,**kwargs):
        file_path = Path(file_path)
        ds = load_dataset(str(file_path.parent),
                                  data_files=file_path.name,
                                  split='train')
        return TextDataController(ds,**kwargs)
        
    
    @classmethod
    def from_df(cls,df,validate=True,**kwargs):
        if validate:
            check_input_validation(df)
        ds = Dataset.from_pandas(df)
        return TextDataController(ds,**kwargs)
    
    @classmethod
    def from_pickle(cls,
                    fname, # Name of the pickle file
                    parent='pickle_files' # Parent folder
                   ):
        return load_pickle(fname,parent=parent)
    
    def _determine_multihead_multilabel(self):
        self.is_multilabel=False
        self.is_multihead=False
        if self.label_names is None: return
        
        if len(self.label_names)>1:
            self.is_multihead=True
        # get label of first row
        first_label = self.dset[self.label_names[0]][0] if not self.is_streamed else next(iter(self.dset))[self.label_names[0]]
        if isinstance(first_label,(list,set,tuple)):
            # This is multi-label. Ignore self.label_names[1:]
            self.label_names = [self.label_names[0]]
            self.is_multihead=False
            self.is_multilabel=True
            
            
    def _map_dset(self,dset,func,is_batched=None,batch_size=None,num_proc=None):
        if is_batched is None: is_batched = self.is_batched
        if batch_size is None: batch_size = self.batch_size
        if num_proc is None: num_proc = self.num_proc
        if self.is_streamed:
            return dset.map(func,
                            batched=is_batched,
                            batch_size=batch_size
                           )
        return dset.map(func,
                        batched=is_batched,
                        batch_size=batch_size,
                        num_proc=num_proc
                       )
    
    def _filter_dset(self,dset,func):
        if self.is_streamed:
            return dset.filter(func,
                            batched=self.is_batched,
                            batch_size=self.batch_size
                           )
        return dset.filter(func,
                        batched=self.is_batched,
                        batch_size=self.batch_size,
                        num_proc=self.num_proc
                       )
                     
    def validate_input(self):
        if self.is_streamed:
            self.verboseprint('Input validation check is disabled when data is streamed')
            return
        _df = self.dset.to_pandas()
        check_input_validation(_df)
    
    
    
    def save_as_pickles(self,
                        fname, # Name of the pickle file
                        parent='pickle_files', # Parent folder
                        drop_data_attributes=False # Whether to drop all large-size data attributes
                       ):
        if drop_data_attributes:
            if hasattr(self, 'main_ddict'):
                del self.main_ddict
        save_to_pickle(self,fname,parent=parent)
    
        
    def _check_validation_leaking(self):
        if self.val_ratio is None or self.is_streamed:
            return
        
        trn_txt = self.main_ddict['train'][self.main_text]
        val_txt = self.main_ddict['validation'][self.main_text]        
        val_txt_leaked = check_text_leaking(trn_txt,val_txt)
        
        if len(val_txt_leaked)==0: return
        
        # filter train dataset to get rid of leaks
        self.verboseprint('Filtering leaked data out of training set...')
        _func = partial(lambda_batch,
                        feature=self.main_text,
                        func=lambda x: x.strip().lower() not in val_txt_leaked,
                        is_batched=self.is_batched)
        self.main_ddict['train'] = self._filter_dset(self.main_ddict['train'],_func)   
        self.verboseprint('Done')
           
    def _train_test_split(self):
        print_msg('Train Test Split',20,verbose=self.verbose)
        val_key = list(set(self.ddict_rest.keys()) & set(['val','validation','valid']))
        if len(val_key)==1: # val split exists
            self.verboseprint('Validation split already exists')
            self.main_ddict=DatasetDict({'train':self.dset,
                                         'validation':self.ddict_rest.pop(val_key[0])})
            
    
        elif self.val_ratio is None: # use all data
            self.verboseprint('No validation split defined')
            self.main_ddict=DatasetDict({'train':self.dset})
        
        elif isinstance(self.val_ratio,list) or isinstance(self.val_ratio,np.ndarray): # filter with indices
            self.verboseprint('Validation indices are provided')
            if self.is_streamed: raise ValueError('Data streaming does not support validation set filtering using indices')
            val_idxs = list(self.val_ratio)
            trn_idxs = list(set(range(len(self.dset))) - set(val_idxs))
            self.main_ddict=DatasetDict({'train':self.dset.select(trn_idxs),
                                         'validation':self.dset.select(val_idxs)})
            
        elif (isinstance(self.val_ratio,float) or isinstance(self.val_ratio,int)) and not len(self.stratify_cols):
            self.verboseprint('Validation split based on val_ratio')
            if self.is_streamed:
                # shuffle dataset before splitting it. This is memory-consuming
#                 self.dset = self.dset.shuffle(seed=self.seed,buffer_size=self.buffer_size)
                if isinstance(self.val_ratio,float):
                    warnings.warn("Length of streamed dataset is unknown to use float validation ratio. Default to the first 1000 data points for validation")
                    self.val_ratio=1000  
                trn_dset = self.dset.skip(self.val_ratio)  
                val_datas = list(self.dset.take(self.val_ratio))
                val_dict={k: [v[k] for v in val_datas] for k in val_datas[0].keys()}   
                val_dset = Dataset.from_dict(val_dict) 
                self.main_ddict=DatasetDict({'train':trn_dset,
                                         'validation':val_dset})
            else:
                # train val split
                self.main_ddict = self.dset.train_test_split(test_size=self.val_ratio,shuffle=True,seed=self.seed)
                self.main_ddict['validation']=self.main_ddict['test']
                del self.main_ddict['test']
        
        else: # val_ratio split with stratifying
            if self.is_streamed: raise ValueError('Stratified split is not supported for streamed data')                
            if self.is_multilabel and self.label_names[0] in self.stratify_cols:
                raise ValueError('For MultiLabel classification, you cannot choose the label as your stratified column')
            self.verboseprint('Validation split based on val_ratio, with stratifying')
            # Create a new feature 'stratified', which is a concatenation of values in stratify_cols
            if self.is_batched:
                stratified_creation = lambda x: {'stratified':
                                     ['_'.join(list(map(str,[x[v][i] for v in self.stratify_cols]))) 
                                      for i in range(len(x[self.stratify_cols[0]]))]}
            else:
                stratified_creation = lambda x: {'stratified':
                                     '_'.join(list(map(str,[x[v] for v in self.stratify_cols]))) 
                                      }
            self.dset = self.dset.map(stratified_creation,
                                      batched=self.is_batched,
                                      batch_size=self.batch_size,
                                      num_proc=self.num_proc)
            self.dset=self.dset.class_encode_column("stratified")
            # train val split
            self.main_ddict = self.dset.train_test_split(test_size=self.val_ratio,
                                                         shuffle=True,seed=self.seed,
                                                        stratify_by_column='stratified')
            self.main_ddict['validation']=self.main_ddict['test']
            del self.main_ddict['test']
            self.main_ddict=self.main_ddict.remove_columns(['stratified'])
            
        
        del self.dset
        self.verboseprint('Done')

                             
    def _create_label_mapping_func(self,encoder_classes):
        if self.is_multihead:
            label2idxs = [{v:i for i,v in enumerate(l_classes)} for l_classes in encoder_classes]
                    
            _func = lambda inp: {'label': [[label2idxs[i][v] for i,v in enumerate(vs)] for vs in zip(*[inp[l] for l in self.label_names])] \
                                    if self.is_batched else [label2idxs[i][v] for i,v in enumerate([inp[l] for l in self.label_names])]
                              }
            
        else:
            label2idx = {v:i for i,v in enumerate(encoder_classes[0])}
            _func = partial(lambda_map_batch,
                           feature=self.label_names[0],
                           func=lambda x: label2idx[x],
                           output_feature='label',
                           is_batched=self.is_batched)
        return _func
        
    def _encode_labels(self):
        if self.label_names is None: return
        print_msg('Label Encoding',verbose=self.verbose)
        
        if self.label_lists is not None and not isinstance(self.label_lists[0],list):
            self.label_lists = [self.label_lists]
                    
        encoder_classes=[]
        if not self.is_multilabel:
            for idx,l in enumerate(self.label_names):
                if self.label_lists is None:
                    l_encoder = LabelEncoder()
                    _ = l_encoder.fit(self.dset[l])
                    l_classes = list(l_encoder.classes_)
                else:
                    l_classes = sorted(list(self.label_lists[idx]))
                encoder_classes.append(l_classes)
            
            _func = self._create_label_mapping_func(encoder_classes)
                
            self.dset = self._map_dset(self.dset,_func)

            val_key = list(set(self.ddict_rest.keys()) & set(['val','validation','valid']))
            if len(val_key)>1: raise ValueError('Your DatasetDict has more than 1 validation split')
            if len(val_key)==1:
                val_key=val_key[0]
                self.ddict_rest[val_key] = self._map_dset(self.ddict_rest[val_key],_func)
                    
        else:
            # For MultiLabel, we transform the label itself to one-hot (or actually, few-hot)
            if self.label_lists is None:
                l_encoder = MultiLabelBinarizer()
                _ = l_encoder.fit(self.dset[self.label_names[0]])
                l_classes = list(l_encoder.classes_)
            else:
                l_classes = sorted(list(self.label_lists[0]))
            
            encoder_classes.append(l_classes)
            
            l_encoder = MultiLabelBinarizer(classes=encoder_classes[0])
            _ = l_encoder.fit(None)
            _func = partial(lambda_map_batch,
                            feature=self.label_names[0],
                            func=lambda x: l_encoder.transform(x),
                            output_feature='label',
                            is_batched=self.is_batched,
                            is_func_batched=True)
            self.dset = self._map_dset(self.dset,_func)                                                  
            
            val_key = list(set(self.ddict_rest.keys()) & set(['val','validation','valid']))
            if len(val_key)>1: raise ValueError('Your DatasetDict has more than 1 validation dataset')
            if len(val_key)==1:
                val_key=val_key[0]
                self.ddict_rest[val_key] = self._map_dset(self.ddict_rest[val_key],_func)
            
        self.label_lists = encoder_classes
        self.verboseprint('Done')
        
    def _process_metadatas(self,dset,ddict_rest=None):
        if len(self.metadatas)>0:
            print_msg('Metadata Simple Processing & Concatenating to Main Content',verbose=self.verbose)
            map_func = partial(concat_metadatas,
                               main_text=self.main_text,
                               metadatas=self.metadatas,
                               process_metas=self.process_metas,
                               is_batched=self.is_batched)
            dset = self._map_dset(dset,map_func)
            if ddict_rest is not None:
                ddict_rest = self._map_dset(ddict_rest,map_func)
            self.verboseprint('Done')
        return dset if ddict_rest is None else (dset,ddict_rest)
            
            
    
    def _simplify_ddict(self):
        print_msg('Dropping unused features',20,verbose=self.verbose)
        if self.cols_to_keep is None:
            self.cols_to_keep= [self.main_text] + self.metadatas
            if self.label_names is not None: self.cols_to_keep+=self.label_names
        cols_to_remove = set(self.all_cols) - set(self.cols_to_keep)
        self.main_ddict['train']=self.main_ddict['train'].remove_columns(list(cols_to_remove))
        if 'validation' in self.main_ddict.keys():
            self.main_ddict['validation']=self.main_ddict['validation'].remove_columns(list(cols_to_remove))
        self.verboseprint('Done')
    
    def _do_transformation(self,dset,ddict_rest=None):
        if len(self.content_tfms):
            print_msg('Text Transformation',20,verbose=self.verbose)
            for tfm in self.content_tfms:
                print_msg(callable_name(tfm),verbose=self.verbose)
                _func = partial(lambda_map_batch,
                               feature=self.main_text,
                               func=tfm,
                               is_batched=self.is_batched)
                dset = self._map_dset(dset,_func)
                if ddict_rest is not None:
                    ddict_rest = self._map_dset(ddict_rest,_func)
            self.verboseprint('Done')
        return dset if ddict_rest is None else (dset,ddict_rest)
 
    def _do_filtering(self,dset,ddict_rest=None):
        if len(self.filter_dict):
            print_msg('Data Filtering',20,verbose=self.verbose)
            col_names = get_dset_col_names(dset)
            for f,tfm in self.filter_dict.items():
                if f in col_names:
                    print_msg(f'Do {callable_name(tfm)} on {f}',verbose=self.verbose)
                    _func = partial(lambda_batch,
                                    feature=f,
                                    func=tfm,
                                    is_batched=self.is_batched)
                    dset = self._filter_dset(dset,_func)
                if ddict_rest is not None:
                    ddict_rest = self._filter_dset(ddict_rest,_func)
            self.verboseprint('Done')
        return dset if ddict_rest is None else (dset,ddict_rest)
    
    def _upsampling(self):
        if len(self.upsampling_list):
            print_msg('Upsampling data',20,verbose=self.verbose)
            results=[]
            for f,tfm in self.upsampling_list:
                print_msg(f'Do {callable_name(tfm)} on {f}',verbose=self.verbose)
                _func = partial(lambda_batch,
                                feature=f,
                                func=tfm,
                                is_batched=self.is_batched)
                new_dset = self._filter_dset(self.main_ddict['train'],_func)
                results.append(new_dset)
            # slow concatenation for iterable dataset    
            self.main_ddict['train'] = concatenate_datasets(results+[self.main_ddict['train']])
            self.verboseprint('Done')
      
    def _do_augmentation(self):
        
        if len(self.aug_tfms):
            print_msg('Text Augmentation',20,verbose=self.verbose)

            seed_notorch(self.seed)
            if not self.is_streamed:  
#                 self.main_ddict['train'] = self.main_ddict['train'].with_transform(partial(augmentation_helper,
#                                                                        text_name=self.main_text,
#                                                                        func=partial(func_all,functions=self.aug_tfms)))              
                for tfm in self.aug_tfms:
                    print_msg(callable_name(tfm),verbose=self.verbose)
            
                    bs = self.batch_size
                    is_func_batched=False
                    num_proc = self.num_proc
                    is_batched = self.is_batched
                    if hasattr(tfm, "run_on_gpu") and getattr(tfm,'run_on_gpu')==True:
                        bs = 32 if not hasattr(tfm, "batch_size") else getattr(tfm,'batch_size')
                        is_func_batched=True
                        is_batched=True
                        num_proc=1
                        
                    _func = partial(lambda_map_batch,
                                   feature=self.main_text,
                                   func=tfm,
                                   is_batched=is_batched,
                                   is_func_batched=is_func_batched
                                   )
                    self.main_ddict['train'] = self._map_dset(self.main_ddict['train'],_func,
                                                              is_batched=is_batched,
                                                              batch_size=bs,
                                                              num_proc=num_proc
                                                             )

            else:
                self.main_ddict['train'] = IterableDataset.from_generator(augmentation_stream_generator,
                                               features = self.main_ddict['train'].features,
                                               gen_kwargs={'dset': self.main_ddict['train'],
                                                           'text_name':self.main_text,
                                                           'func':partial(func_all,functions=self.aug_tfms)
                                                          })
            self.verboseprint('Done')
        
    def _convert_to_iterable(self):
        if (not self.is_streamed) and self.convert_training_to_iterable:
            print_msg('Converting train set to iterable',20,verbose=self.verbose)
            self.main_ddict['train'] = self.main_ddict['train'].to_iterable_dataset(num_shards=self.num_shards)
            self.is_streamed=True
            self.verboseprint('Done')

            
    def _do_train_shuffling(self):
        print_msg('Shuffling train set',20,verbose=self.verbose)
        self.main_ddict['train'] = self.main_ddict['train'].shuffle(seed=self.seed, buffer_size=self.buffer_size)
        self.verboseprint('Done')
        
    def do_all_preprocessing(self,
                             shuffle_trn=True # To shuffle the train set before tokenization
                            ):
        if self._processed_call:
            warnings.warn('Your dataset has already been processed. Returning the previous processed DatasetDict...')
            return self.main_ddict
            
        print_msg('Start Main Text Processing',20,verbose=self.verbose)
        
        # Filtering
        self.dset,self.ddict_rest = self._do_filtering(self.dset,self.ddict_rest)
        
        # Process metadatas
        self.dset,self.ddict_rest = self._process_metadatas(self.dset,self.ddict_rest)
        
        # Process labels
        self._encode_labels()
        
        # Content transformation
        self.dset,self.ddict_rest = self._do_transformation(self.dset,self.ddict_rest)
         
        # Train Test Split.
        ### self.main_ddict is created here
        self._train_test_split()
        
        # Dropping unused columns
        self._simplify_ddict()
        
        # Check validation leaking
        self._check_validation_leaking()
        
        ### The rest of these functions applies only to the train dataset
        # Upsampling
        self._upsampling()
        
        # Augmentation
        self._do_augmentation()
           
        # Convert train set to iterable
        self._convert_to_iterable()
        
        # Shuffle train
        if shuffle_trn:
            self._do_train_shuffling()
        
        self._processed_call=True
        
        return self.main_ddict
    
        
    def do_tokenization(self,
                             tokenizer, # Tokenizer (preferably from HuggingFace)
                             is_split_into_words=False, # Is text split into list or not
                             max_length=None, # pad to model's allowed max length (default is max_sequence_length)
                             trn_size=None, # The number of training data to be tokenized
                            ):
        print_msg('Tokenization',20,verbose=self.verbose)
        self.tokenizer = tokenizer
        self.is_split_into_words= is_split_into_words
        self.max_length = max_length
        if trn_size is not None:
            self.main_ddict['train'] = self.main_ddict['train'].take(trn_size)
        
        for k in self.main_ddict.keys():
            self.main_ddict[k] = self.main_ddict[k].map(partial(tokenize_function,
                                                                text_name=self.main_text,
                                                                tok=tokenizer,
                                                                is_split_into_words=is_split_into_words,
                                                                max_length=max_length),
                                                            batched=True,
                                                            batch_size=self.batch_size
                                                           )
        self.verboseprint('Done')
        return self.main_ddict
        
    def process_and_tokenize(self,
                             tokenizer, # Tokenizer (preferably from HuggingFace)
                             is_split_into_words=False, # Is text split into list or not
                             max_length=None, # pad to model's allowed max length (default is max_sequence_length)
                             trn_size=None, # The number of training data to be tokenized
                             shuffle_trn=True, # To shuffle the train set before tokenization
                            ):
        """
        This will perform `do_all_processing` then `do_tokenization`
        """
        _ = self.do_all_preprocessing(shuffle_trn)
        _ = self.do_tokenization(tokenizer,is_split_into_words,max_length,trn_size)
        
    
    def set_data_collator(self,data_collator):
        self.data_collator = data_collator
        
    
    def prepare_test_dataset_from_csv(self,
                                      file_path, # path to csv file
                                      do_filtering=False # whether to perform data filtering on this test set
                                     ):
        file_path = Path(file_path)
        ds = load_dataset(str(file_path.parent),
                          data_files=file_path.name,
                          split='train')
        return self.prepare_test_dataset(ds,do_filtering)
    
    def prepare_test_dataset_from_df(self,
                                     df, # Pandas Dataframe
                                     validate=True, # whether to perform input data validation
                                     do_filtering=False # whether to perform data filtering on this test set 
                                    ):
        if validate:
            check_input_validation(df)
        ds = Dataset.from_pandas(df)
        return self.prepare_test_dataset(ds,do_filtering)
        
    def prepare_test_dataset(self,
                             test_dset, # The HuggingFace Dataset as Test set
                             do_filtering=False # whether to perform data filtering on this test set
                            ):
        test_cols = set(get_dset_col_names(test_dset))
        test_cols = test_cols - set(self.label_names)
        missing_cols = set(self.cols_to_keep) - set(self.label_names) - set(test_cols)
        if len(missing_cols):
            raise ValueError(f'Test set does not have these columns required for preprocessings: {missing_cols}')
            
        print_msg('Start Test Set Transformation',20,verbose=self.verbose)

        # Filtering
        if do_filtering:
            test_dset = self._do_filtering(test_dset)
        
        # Process metadatas
        test_dset = self._process_metadatas(test_dset)
        
        # Content transformation
        test_dset = self._do_transformation(test_dset)
        
        # Drop unused columns
        cols_to_remove = test_cols - set(self.cols_to_keep)
        test_dset=test_dset.remove_columns(list(cols_to_remove))
        
        # Tokenization
        print_msg('Tokenization',20,verbose=self.verbose)
        test_dset = test_dset.map(partial(tokenize_function,
                                          text_name=self.main_text,
                                          tok=self.tokenizer,
                                          is_split_into_words=self.is_split_into_words,
                                          max_length=self.max_length),
                                  batched=True,
                                  batch_size=self.batch_size
                                 )
        self.verboseprint('Done')
        return test_dset


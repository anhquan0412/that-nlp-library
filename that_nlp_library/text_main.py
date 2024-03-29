# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_text_main.ipynb.

# %% ../nbs/00_text_main.ipynb 3
from __future__ import annotations
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from datasets import DatasetDict,Dataset,IterableDataset,load_dataset,concatenate_datasets,Value
from pathlib import Path
from .utils import *
from functools import partial
import warnings
from datasets.utils.logging import disable_progress_bar,enable_progress_bar

# %% auto 0
__all__ = ['tokenizer_explain', 'two_steps_tokenization_explain', 'tokenize_function', 'concat_metadatas', 'TextDataController']

# %% ../nbs/00_text_main.ipynb 6
def tokenizer_explain(inp, # Input sentence
                      tokenizer, # Tokenizer (preferably from HuggingFace)
                      split_word=False # Is input `inp` split into list or not
                     ):
    "Display results from tokenizer"
    print('\t\t------- Tokenizer Explained -------')
    print('----- Input -----')
    print(inp)
    print()
    print('----- Tokenized results ----- ')
    print(tokenizer(inp,is_split_into_words=split_word))
    print()
    tok = tokenizer.encode(inp,is_split_into_words=split_word)
    print('----- Results from tokenizer.convert_ids_to_tokens -----')
    print(tokenizer.convert_ids_to_tokens(tok))
    print()
    print('----- Results from tokenizer.decode ----- ')
    print(tokenizer.decode(tok))
    print()

# %% ../nbs/00_text_main.ipynb 13
def two_steps_tokenization_explain(inp, # Input sentence
                                   tokenizer, # Tokenizer (preferably from HuggingFace)
                                   content_tfms=[], # A list of text transformations
                                   aug_tfms=[], # A list of text augmentation 
                                  ):
    "Display results form each content transformation, then display results from tokenizer"
    print('\t\t------- Text Transformation Explained -------')
    print('----- Raw sentence -----')
    print(inp)
    print()
    print('----- Content Transformations (on both train and test) -----')
    content_tfms = val2iterable(content_tfms)
    for tfm in content_tfms:
        print_msg(callable_name(tfm),3)
        inp = tfm(inp)
        print(inp)
        print()
    print()
    print('----- Augmentations (on train only) -----')
    aug_tfms = val2iterable(aug_tfms)
    for tfm in aug_tfms:
        print_msg(callable_name(tfm),3)
        inp = tfm(inp)
        print(inp)
        print()
    print()
    tokenizer_explain(inp,tokenizer)

# %% ../nbs/00_text_main.ipynb 34
def tokenize_function(text,
                      tok,
                      max_length=None,
                      is_split_into_words=False,
                      return_tensors=None,
                      return_special_tokens_mask=False
                     ):
    if max_length is None:
        # pad to model's default max sequence length
        return tok(text, padding="max_length", 
                   truncation=True,
                   is_split_into_words=is_split_into_words,
                   return_tensors=return_tensors,
                   return_special_tokens_mask=return_special_tokens_mask
                  )
    if isinstance(max_length,int) and max_length>0:
        # pad to the largest length of the current batch, and start truncating at max_length
        return tok(text, padding=True, 
                   max_length=max_length,
                   truncation=True,
                   is_split_into_words=is_split_into_words,
                   return_tensors=return_tensors,
                   return_special_tokens_mask=return_special_tokens_mask
                  )
    
    # no padding (still truncate at model's default max sequence length)
    return tok(text, padding=False,
               truncation=True,
               is_split_into_words=is_split_into_words,
               return_tensors=return_tensors,
               return_special_tokens_mask=return_special_tokens_mask
              )

# %% ../nbs/00_text_main.ipynb 51
def concat_metadatas(dset:dict, # HuggingFace Dataset
                     main_text, # Text feature name
                     metadatas, # Metadata (or a list of metadatas)
                     process_metas=True, # Whether apply simple metadata processing, i.e. space strip and lowercase
                     sep='.', # Separator, for multiple metadatas concatenation
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
            m_data = [none2emptystr(v).strip().lower() for v in m_data] if is_batched else none2emptystr(m_data).strip().lower()
        results[m]=m_data
        if is_batched:
            results[main_text] = [f'{m_data[i]} {sep} {results[main_text][i]}' for i in range(len(m_data))]
        else:
            results[main_text] = f'{m_data} {sep} {results[main_text]}'
    return results

# %% ../nbs/00_text_main.ipynb 54
class TextDataController():
    def __init__(self,
                 inp, # HuggingFainpce Dataset or DatasetDict
                 main_text:str, # Name of the main text column
                 label_names=[], # Names of the label (dependent variable) columns
                 sup_types=[], # Type of supervised learning for each label name ('classification' or 'regression')
                 class_names_predefined=[], # List of names associated with the labels (same index order). Use empty list for regression
                 filter_dict={}, # A dictionary: {feature: filtering_function_for_that_feature}
                 label_tfm_dict={}, # A dictionary: {label_name: transform_function_for_that_label}
                 metadatas=[], # Names of the metadata columns
                 process_metas=True, # Whether to do simple text processing on the chosen metadatas
                 metas_sep='.', # Separator, for multiple metadatas concatenation
                 content_transformations=[], # A list of text transformations
                 val_ratio:int|float|None=0.2, # Ratio of data for validation set
                 stratify_cols=[], # Column(s) needed to do stratified shuffle split
                 upsampling_list={}, # A list of tuple. Each tuple: (feature,upsampling_function_based_on_the_feature)
                 content_augmentations=[], # A list of text augmentations
                 seed=None, # Random seed
                 batch_size=1024, # CPU batch size
                 num_proc=4, # Number of processes for multiprocessing
                 cols_to_keep=None, # Columns to keep after all processings
                 verbose=True, # Whether to print processing information
                ):
            
        self.main_text = main_text
        
        self.label_names = val2iterable(label_names)
        self.sup_types = val2iterable(sup_types)
        self._check_sup_types()
        self.label_lists = class_names_predefined
        
        self.filter_dict = filter_dict
        self.label_tfm_dict = label_tfm_dict
        self.metadatas = val2iterable(metadatas)
        self.process_metas = process_metas
        self.metas_sep = metas_sep
        
        self.content_tfms = val2iterable(content_transformations)
        
        self.val_ratio = val_ratio
        self.stratify_cols = val2iterable(stratify_cols)
        
        self.upsampling_list = upsampling_list
        self.aug_tfms = val2iterable(content_augmentations)
        
        self.seed = seed
        self.is_batched = batch_size>1
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.cols_to_keep = cols_to_keep
        
        self.ddict_rest = DatasetDict()
        self.set_verbose(verbose)
        
        if hasattr(inp,'keys'): # is datasetdict
            if 'train' in inp.keys(): 
                self.ddict_rest = inp
                self.dset = self.ddict_rest.pop('train')
            else:
                raise ValueError('The given DatasetDict has no "train" split')
        else: # is dataset
            self.dset = inp
        
        if isinstance(self.dset,IterableDataset):
            raise ValueError('TextDataController does not handle Iterable Dataset as input. Consider `TextDataControllerStreaming`')
        
        self._determine_val_key()
        self.all_cols = get_dset_col_names(self.dset)
        self._determine_multihead_multilabel()
        self._convert_regression_to_float()
        
        self._processed_call=False
            
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

    def set_verbose(self,verbose):
        self.verbose = verbose
        self.verboseprint = print if verbose else lambda *a, **k: None
        if not self.verbose:
            disable_progress_bar() # turn off huggingface `map` progress bar
        else:
            enable_progress_bar()
    
    def _convert_regression_to_float(self):
        if len(self.sup_types)==0: return
        # convert regression labels to float64
        reg_idxs = [i for i,v in enumerate(self.sup_types) if v=='regression']
        for i in reg_idxs:
            self.dset = self.dset.cast_column(self.label_names[i],Value("float64"))
            if self.val_key is not None:
                self.ddict_rest[self.val_key] = self.ddict_rest[self.val_key].cast_column(self.label_names[i],Value("float64"))
        
    def _check_sup_types(self):
        assert len(self.label_names)==len(self.sup_types), "The number of supervised learning declaration must equal to the number of label"
        assert len(set(self.sup_types) - set(['classification','regression']))==0, 'Accepted inputs for `sup_types` are `classification` and `regression`'
    
    def _determine_val_key(self):
        # determine validation split key
        self.val_key=None
        val_key = list(set(self.ddict_rest.keys()) & set(['val','validation','valid']))
        if len(val_key)>1: raise ValueError('Your DatasetDict has more than 1 validation split')
        if len(val_key)==1:
            val_key=val_key[0]
            self.val_key=val_key
            
    def _determine_multihead_multilabel(self):
        self.is_multilabel=False
        self.is_multihead=False
        if len(self.label_names)==0: return
        
        if len(self.label_names)>1:
            self.is_multihead=True
            
        # get label of first row
        first_label = self.dset[self.label_names[0]][0]
        if isinstance(first_label,(list,set,tuple)):
            # This is multi-label. Ignore self.label_names[1:]
            self.label_names = [self.label_names[0]]
            self.is_multihead=False
            self.is_multilabel=True
                     
    def validate_input(self):
        _df = self.dset.to_pandas()
        check_input_validation(_df)
    
    def save_as_pickles(self,
                        fname, # Name of the pickle file
                        parent='pickle_files', # Parent folder
                        drop_attributes=False # Whether to drop large-size attributes
                       ):
        if drop_attributes:
            if hasattr(self, 'main_ddict'):
                del self.main_ddict
            if hasattr(self, 'ddict_rest'):
                del self.ddict_rest
            if hasattr(self, 'aug_tfms'):
                del self.aug_tfms
        save_to_pickle(self,fname,parent=parent)
    
        
    def _check_validation_leaking(self):
        if self.val_ratio is None:
            return
        
        trn_txt = self.main_ddict['train'][self.main_text]
        val_txt = self.main_ddict['validation'][self.main_text]        
        val_txt_leaked = check_text_leaking(trn_txt,val_txt,verbose=self.verbose)
        
        if len(val_txt_leaked)==0: return
        
        # filter train dataset to get rid of leaks
        self.verboseprint('Filtering leaked data out of training set...')
        _func = partial(lambda_batch,
                        feature=self.main_text,
                        func=lambda x: x.strip() not in val_txt_leaked,
                        is_batched=self.is_batched)
        self.main_ddict['train'] = hf_filter_dset(self.main_ddict['train'],_func,self.is_batched,self.batch_size,self.num_proc)   
        self.verboseprint('Done')
    
    def _process_metadatas(self,dset,ddict_rest=None):
        if len(self.metadatas):
            print_msg('Metadata Simple Processing & Concatenating to Main Content',verbose=self.verbose)
            map_func = partial(concat_metadatas,
                               main_text=self.main_text,
                               metadatas=self.metadatas,
                               process_metas=self.process_metas,
                               sep=self.metas_sep,
                               is_batched=self.is_batched)
            dset = hf_map_dset(dset,map_func,self.is_batched,self.batch_size,self.num_proc)
            if ddict_rest is not None:
                ddict_rest = hf_map_dset(ddict_rest,map_func,self.is_batched,self.batch_size,self.num_proc)
            self.verboseprint('Done')
        return dset if ddict_rest is None else (dset,ddict_rest)
    
    def _do_label_transformation(self):
        if len(self.label_names)==0 or len(self.label_tfm_dict)==0: return
        print_msg('Label Transformation',20,verbose=self.verbose)
        for f,tfm in self.label_tfm_dict.items():
            if f in self.label_names:
                _func = partial(lambda_map_batch,
                                feature=f,
                                func=tfm,
                                is_batched=self.is_batched
                               )                
                self.dset = hf_map_dset(self.dset,_func,self.is_batched,self.batch_size,self.num_proc)
                if self.val_key is not None:
                    self.ddict_rest[self.val_key] = hf_map_dset(self.ddict_rest[self.val_key],
                                                                _func,
                                                                self.is_batched,
                                                                self.batch_size,
                                                                self.num_proc)
        self.verboseprint('Done')
        
    def _create_label_mapping_func(self,encoder_classes):
        if self.is_multihead:
            label2idxs = [{v:i for i,v in enumerate(l_classes)} for l_classes in encoder_classes]
            _func = lambda inp: {'label': [[label2idxs[i][v] if len(label2idxs[i]) else v for i,v in enumerate(vs)] \
                                           for vs in zip(*[inp[l] for l in self.label_names])] if self.is_batched \
                                 else [label2idxs[i][v] if len(label2idxs[i]) else v for i,v in enumerate([inp[l] for l in self.label_names])]
                                }
            
        else: # single-head
            if self.sup_types[0]=='regression':
                _func1 = lambda x: x
            else:
                label2idx = {v:i for i,v in enumerate(encoder_classes[0])}
                _func1 = lambda x: label2idx[x]
                
            _func = partial(lambda_map_batch,
                           feature=self.label_names[0],
                           func=_func1,
                           output_feature='label',
                           is_batched=self.is_batched)
        return _func
        
    def _encode_labels(self):
        if len(self.label_names)==0: return
        print_msg('Label Encoding',verbose=self.verbose)
        
        if len(self.label_lists) and not isinstance(self.label_lists[0],list):
            self.label_lists = [self.label_lists]
                    
        encoder_classes=[]
        if not self.is_multilabel:
            for idx,l in enumerate(self.label_names):
                if self.sup_types[idx]=='regression':
                    l_classes=[]
                else:
                    # classification
                    if len(self.label_lists)==0:
                        l_encoder = LabelEncoder()
                        _ = l_encoder.fit(self.dset[l])
                        l_classes = list(l_encoder.classes_)
                    else:
                        l_classes = sorted(list(self.label_lists[idx]))
                encoder_classes.append(l_classes)
            
            _func = self._create_label_mapping_func(encoder_classes)
                
            self.dset = hf_map_dset(self.dset,_func,self.is_batched,self.batch_size,self.num_proc)
            if self.val_key is not None:
                self.ddict_rest[self.val_key] = hf_map_dset(self.ddict_rest[self.val_key],_func,
                                                       self.is_batched,self.batch_size,self.num_proc)
                    
        else:
            # For MultiLabel, we transform the label itself to one-hot (or actually, few-hot)
            if len(self.label_lists)==0:
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
            self.dset = hf_map_dset(self.dset,_func,self.is_batched,self.batch_size,self.num_proc)                                                 
            
            if self.val_key is not None:
                self.ddict_rest[self.val_key] = hf_map_dset(self.ddict_rest[self.val_key],_func,
                                                       self.is_batched,self.batch_size,self.num_proc)
            
        self.label_lists = encoder_classes
        self.verboseprint('Done')
        
    
    def _train_test_split(self):
        print_msg('Train Test Split',20,verbose=self.verbose)
        if self.val_key is not None: # val split exists
            self.verboseprint('Validation split already exists')
            self.main_ddict=DatasetDict({'train':self.dset,
                                         'validation':self.ddict_rest.pop(self.val_key)})
            
    
        elif self.val_ratio is None: # use all data
            self.verboseprint('No validation split defined')
            self.main_ddict=DatasetDict({'train':self.dset})
            
        elif (isinstance(self.val_ratio,float) or isinstance(self.val_ratio,int)) and len(self.stratify_cols)==0:
            self.verboseprint('Validation split based on val_ratio')
            # train val split
            self.main_ddict = self.dset.train_test_split(test_size=self.val_ratio,shuffle=True,seed=self.seed)
            self.main_ddict['validation']=self.main_ddict['test']
            del self.main_ddict['test']
        
        else: # val_ratio split with stratifying             
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
    
    def _simplify_ddict(self):
        print_msg('Dropping unused features',20,verbose=self.verbose)
        if self.cols_to_keep is None:
            self.cols_to_keep= [self.main_text] + self.metadatas + self.label_names
            
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
                dset = hf_map_dset(dset,_func,self.is_batched,self.batch_size,self.num_proc)
                if ddict_rest is not None:
                    ddict_rest = hf_map_dset(ddict_rest,_func,self.is_batched,self.batch_size,self.num_proc)
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
                    dset = hf_filter_dset(dset,_func,self.is_batched,self.batch_size,self.num_proc)
                if ddict_rest is not None: # assuming ddict_rest has the column to filter, always
                    ddict_rest = hf_filter_dset(ddict_rest,_func,self.is_batched,self.batch_size,self.num_proc)
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
                new_dset = hf_filter_dset(self.main_ddict['train'],_func,self.is_batched,self.batch_size,self.num_proc)
                results.append(new_dset)
            # slow concatenation for iterable dataset    
            self.main_ddict['train'] = concatenate_datasets(results+[self.main_ddict['train']])
            self.verboseprint('Done')
      
    def _do_augmentation(self):
        if len(self.aug_tfms):
            print_msg('Text Augmentation',20,verbose=self.verbose)
            for tfm in self.aug_tfms:
                print_msg(callable_name(tfm),verbose=self.verbose)
                bs = self.batch_size
                is_func_batched=False
                num_proc = self.num_proc
                is_batched = self.is_batched
                if hasattr(tfm, "run_on_gpu") and getattr(tfm,'run_on_gpu')==True:
                    bs = min(32,self.batch_size) if not hasattr(tfm, "batch_size") else getattr(tfm,'batch_size')
                    is_func_batched=True
                    is_batched=True
                    num_proc=1

                _func = partial(lambda_map_batch,
                               feature=self.main_text,
                               func=tfm,
                               is_batched=is_batched,
                               is_func_batched=is_func_batched
                               )
                self.main_ddict['train'] = hf_map_dset(self.main_ddict['train'],_func,
                                                          is_batched=is_batched,
                                                          batch_size=bs,
                                                          num_proc=num_proc
                                                         )


            self.verboseprint('Done')
        

            
    def _do_train_shuffling(self):
        print_msg('Shuffling and flattening train set',20,verbose=self.verbose)
        self.main_ddict['train'] = self.main_ddict['train'].shuffle(seed=self.seed).flatten_indices(num_proc = self.num_proc)
        self.verboseprint('Done')
        
    def do_all_preprocessing(self,
                             shuffle_trn=True, # To shuffle the train set before tokenization
                             check_val_leak=True # To check (and remove) training data which is leaked to validation set 
                            ):
        if self._processed_call:
            warnings.warn('Your dataset has already been processed. Returning the previous processed DatasetDict...')
            return self.main_ddict
            
        print_msg('Start Main Text Processing',20,verbose=self.verbose)
        
        # Filtering
        self.dset,self.ddict_rest = self._do_filtering(self.dset,self.ddict_rest)
        
        # Process metadatas
        self.dset,self.ddict_rest = self._process_metadatas(self.dset,self.ddict_rest)
        
        # Label transformation
        self._do_label_transformation()
        
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
        if check_val_leak:
            self._check_validation_leaking()
        
        ### The rest of these functions applies only to the train dataset
        # Upsampling
        self._upsampling()
        
        # Augmentation
        self._do_augmentation()
        
        # Shuffle train
        if shuffle_trn:
            self._do_train_shuffling()
        
        self._processed_call=True
        
        return self.main_ddict
    
        
    def do_tokenization(self,
                        tokenizer, # Tokenizer (preferably from HuggingFace)
                        max_length=None, # pad to model's allowed max length (default is max_sequence_length). Use -1 for no padding at all
                        trn_size=None, # The number of training data to be tokenized
                        tok_num_proc=None, # Number of processes for tokenization
                       ):
        print_msg('Tokenization',20,verbose=self.verbose)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tok_num_proc = tok_num_proc if tok_num_proc else self.num_proc
        tok_func = partial(tokenize_function,tok=tokenizer,max_length=max_length)
        _func = partial(lambda_map_batch,
                        feature=self.main_text,
                        func=tok_func,
                        output_feature=None,
                        is_batched=self.is_batched)
        
        if trn_size is not None:
            if isinstance(trn_size,float):
                num_shard = int(1/trn_size)
            else: # int
                trn_len=len(self.main_ddict['train'])
                num_shard = trn_len//trn_size
            self.main_ddict['train'] = self.main_ddict['train'].shard(num_shard,0)
        
        for k in self.main_ddict.keys():
            self.main_ddict[k] = hf_map_dset(self.main_ddict[k],_func,self.is_batched,self.batch_size,self.tok_num_proc)

        self.verboseprint('Done')
        return self.main_ddict
        
    def process_and_tokenize(self,
                             tokenizer, # Tokenizer (preferably from HuggingFace)
                             max_length=None, # pad to model's allowed max length (default is max_sequence_length)
                             trn_size=None, # The number of training data to be tokenized
                             tok_num_proc=None, # Number of processes for tokenization
                             shuffle_trn=True, # To shuffle the train set before tokenization
                             check_val_leak=True # To check (and remove) training data which is leaked to validation set
                            ):
        """
        This will perform `do_all_processing` then `do_tokenization`
        """
        if self.seed:
            seed_everything(self.seed) 
        _ = self.do_all_preprocessing(shuffle_trn,check_val_leak)
        _ = self.do_tokenization(tokenizer,max_length,trn_size,tok_num_proc)
        
    
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
    
    def prepare_test_dataset_from_raws(self,
                                       content, # Either a single sentence, list of sentence or a dictionary with keys are metadata columns and values are list
                                      ):
        if len(self.metadatas) and not isinstance(content,dict):
            raise ValueError(f'There is/are metadatas in the preprocessing step. Please include a dictionary including these keys for metadatas: {self.metadatas}, and texture content: {self.main_text}')
            
        _dic = {self.main_text:[content]} if isinstance(content,str) else content
        for k in _dic.keys():
            _dic[k] = val2iterable(_dic[k])
        
        test_dict = Dataset.from_dict(_dic)
        
        # set num_proc to 1 for small data processing
        _tmp1 = self.num_proc
        _tmp2 = self.tok_num_proc
        self.num_proc=1
        self.tok_num_proc=1
        results = self.prepare_test_dataset(test_dict,do_filtering=False)
        self.num_proc = _tmp1
        self.tok_num_proc=_tmp2

        return results
    
    def prepare_test_dataset(self,
                             test_dset, # The HuggingFace Dataset as Test set
                             do_filtering=False, # whether to perform data filtering on this test set
                            ):
        test_cols = set(get_dset_col_names(test_dset))
        label_names_set = set(self.label_names)
        test_cols = test_cols - label_names_set
        missing_cols = set(self.cols_to_keep) - label_names_set - test_cols
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
        _func = partial(lambda_map_batch,
                        feature=self.main_text,
                        func=partial(tokenize_function,tok=self.tokenizer,max_length=self.max_length),
                        output_feature=None,
                        is_batched=self.is_batched)
        test_dset = hf_map_dset(test_dset,_func,self.is_batched,self.batch_size,self.tok_num_proc)
        
        self.verboseprint('Done')
        return test_dset


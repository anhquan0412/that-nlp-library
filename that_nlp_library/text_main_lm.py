# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_text_main_lm.ipynb.

# %% ../nbs/00_text_main_lm.ipynb 3
from __future__ import annotations
from datasets import Dataset
from .utils import *
from .text_main import *
from functools import partial
import warnings
from transformers import DataCollatorForLanguageModeling

# %% auto 0
__all__ = ['TextDataLMController']

# %% ../nbs/00_text_main_lm.ipynb 6
class TextDataLMController(TextDataController):
    def __init__(self,
                 inp, # HuggingFainpce Dataset or DatasetDict
                 main_text:str, # Name of the main text column
                 filter_dict={}, # A dictionary: {feature: filtering_function_for_that_feature}
                 metadatas=[], # Names of the metadata columns
                 process_metas=True, # Whether to do simple text processing on the chosen metadatas
                 content_transformations=[], # A list of text transformations
                 val_ratio:int|float|None=0.2, # Ratio of data for validation set
                 stratify_cols=[], # Column(s) needed to do stratified shuffle split
                 seed=None, # Random seed
                 batch_size=1024, # CPU batch size
                 num_proc=4, # Number of process for multiprocessing
                 cols_to_keep=None, # Columns to keep after all processings
                 verbose=True, # Whether to prdint processing information
                ):
        super().__init__(inp=inp,
                         main_text=main_text,
                         filter_dict=filter_dict,
                         metadatas=metadatas,
                         process_metas=process_metas,
                         content_transformations=content_transformations,
                         val_ratio=val_ratio,
                         stratify_cols=stratify_cols,
                         seed=seed,
                         batch_size=batch_size,
                         num_proc=num_proc,
                         cols_to_keep=cols_to_keep,
                         verbose=verbose
                        )
            
    
    def _do_label_transformation(self):
        raise NotImplementedError("There's no classification/regression label in text processing for Language Model")
        
    def _encode_labels(self):
        raise NotImplementedError("There's no classification/regression label in text processing for Language Model")

    
    def _upsampling(self):
        raise NotImplementedError("There's no upsampling in text processing for Language Model")
      
    def _do_augmentation(self):
        raise NotImplementedError("There's no text augmentation in text processing for Language Model")
     
    def save_as_pickles(self,
                        fname, # Name of the pickle file
                        parent='pickle_files', # Parent folder
                       ):
        
        save_to_pickle(self,fname,parent=parent) 
        
    def _do_train_shuffling(self):
        print_msg('Shuffling and flattening train set',20,verbose=self.verbose)
        self.main_ddict['train'] = self.main_ddict['train'].shuffle(seed=self.seed).flatten_indices(num_proc = self.num_proc)
        self.verboseprint('Done')

    def _group_texts_with_stride(self,examples):
        max_length = self.max_length
        if max_length is None: 
            max_length = self.tokenizer.model_max_length
        stride = self.stride
        if stride is None: stride=max_length
        else: stride = max_length-stride
        if stride==0: raise ValueError(f'Stride cannot be equal to max length of {max_length}')
            
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        result_all={}
        for k,t in concatenated_examples.items():
            result=[]
            i=0
            while i+max_length<=total_length:
                result.append(t[i:i+max_length])
                i+=stride
            result_all[k]=result

        return result_all  
    

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
        
        
        # Content transformation
        self.dset,self.ddict_rest = self._do_transformation(self.dset,self.ddict_rest)
         
        # Train Test Split.
        ### self.main_ddict is created here
        self._train_test_split()
        
        # Dropping unused columns
        self._simplify_ddict()
        
        # Check validation leaking
        self._check_validation_leaking()
        
        # Shuffle train
        if shuffle_trn:
            self._do_train_shuffling()
        
        self._processed_call=True
        
        return self.main_ddict
    
        
    def do_tokenization(self,
                        tokenizer, # Tokenizer (preferably from HuggingFace)
                        max_length=None, # pad to model's allowed max length (default is max_sequence_length). Use -1 for no padding at all
                        line_by_line=True, # To whether tokenize each sentence separately, or concatenate them
                        stride=None, # option to do striding when line_by_line is False
                        trn_size=None, # The number of training data to be tokenized
                        tok_num_proc=None, # Number of processes for tokenization
                       ):
        # References
#         https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
#         https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
        
        print_msg('Tokenization',20,verbose=self.verbose)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_by_line = line_by_line
        self.stride = stride
        self.tok_num_proc = tok_num_proc if tok_num_proc else self.num_proc
        
        tok_func = partial(tokenize_function,tok=self.tokenizer,
                           max_length=max_length if line_by_line else -1,
                           return_special_tokens_mask=True)
        
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
            self.main_ddict[k] = self.main_ddict[k].remove_columns(self.cols_to_keep)
        
        if not line_by_line: # token concatenation
            for k in self.main_ddict.keys():
                self.main_ddict[k] = hf_map_dset(self.main_ddict[k],
                                                 self._group_texts_with_stride,
                                                 is_batched=True,
                                                 batch_size=self.batch_size if self.batch_size>1 else 1024,
                                                 num_proc=self.tok_num_proc)
                
        
        self.verboseprint('Done')
        return self.main_ddict
        
    def process_and_tokenize(self,
                             tokenizer, # Tokenizer (preferably from HuggingFace)
                             max_length=None, # pad to model's allowed max length (default is max_sequence_length)
                             line_by_line=True, # To whether tokenize each sentence separately, or concatenate them and then tokenize
                             stride=None, # option to do striding when line_by_line is False
                             trn_size=None, # The number of training data to be tokenized
                             tok_num_proc=None, # Number of processes for tokenization
                             shuffle_trn=True, # To shuffle the train set before tokenization
                            ):
        """
        This will perform `do_all_processing` then `do_tokenization`
        """
        if self.seed:
            seed_everything(self.seed)
        _ = self.do_all_preprocessing(shuffle_trn)
        _ = self.do_tokenization(tokenizer,max_length,line_by_line,stride,trn_size,tok_num_proc)
        
    
    def set_data_collator(self,
                          is_mlm=True, # Is this masked language model (True) or causal language model (False)
                          mlm_prob=0.15, # Mask probability for masked language model
                         ):
        if not hasattr(self,'max_length'):
            raise ValueError("Please call `process_and_tokenize' or `do_tokenization` to tokenize your dataset")
        
        self.is_mlm = is_mlm
        pad_to_multiple_of_8 = (self.max_length<0) # get data collator to pad
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                             mlm=is_mlm,
                                                             mlm_probability=mlm_prob,
                                                             pad_to_multiple_of=8 if pad_to_multiple_of_8 else None
                                                            )
                                               
    
    def prepare_test_dataset_from_raws(self,
                                       content, # Either a single sentence, list of sentence or a dictionary with keys are metadata columns and values are list
                                       do_tokenize=False, # Whether to tokenize text
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
        results = self.prepare_test_dataset(test_dict,do_tokenize)
        self.num_proc = _tmp1
        self.tok_num_proc=_tmp2
        return results
        
    def prepare_test_dataset(self,
                             test_dset, # The HuggingFace Dataset as Test set
                             do_tokenize, # Whether to tokenize text
                            ):
        test_cols = set(get_dset_col_names(test_dset))
        missing_cols = set(self.cols_to_keep) - test_cols
        if len(missing_cols):
            raise ValueError(f'Test set does not have these columns required for preprocessings: {missing_cols}')
            
        print_msg('Start Test Set Transformation',20,verbose=self.verbose)
        
        # Process metadatas
        test_dset = self._process_metadatas(test_dset)
        
        # Content transformation
        test_dset = self._do_transformation(test_dset)
        
        if not do_tokenize:
            # Drop every columns except for main_text
            cols_to_remove = {c for c in test_cols if c!=self.main_text}
            test_dset = test_dset.remove_columns(list(cols_to_remove))
        else:
            # Drop unused columns
            cols_to_remove = test_cols - set(self.cols_to_keep)
            test_dset = test_dset.remove_columns(list(cols_to_remove))
            
            print_msg('Tokenization',20,verbose=self.verbose)
            tok_func = partial(tokenize_function,
                           tok=self.tokenizer,
                           max_length=self.max_length if self.line_by_line else -1,
                           return_special_tokens_mask=True
                          )
            
            _func = partial(lambda_map_batch,
                        feature=self.main_text,
                        func=tok_func,
                        output_feature=None,
                        is_batched=self.is_batched)
            test_dset = hf_map_dset(test_dset,_func,self.is_batched,self.batch_size,self.tok_num_proc)
            
        self.verboseprint('Done')
        return test_dset

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_text_main_lm_streaming.ipynb.

# %% ../nbs/00_text_main_lm_streaming.ipynb 3
from __future__ import annotations
from datasets import Dataset,IterableDataset
from .utils import *
from .text_main import tokenize_function
from .text_main_streaming import *
from functools import partial
from collections import defaultdict
import warnings
from transformers import DataCollatorForLanguageModeling

# %% auto 0
__all__ = ['TextDataLMControllerStreaming']

# %% ../nbs/00_text_main_lm_streaming.ipynb 6
class TextDataLMControllerStreaming(TextDataControllerStreaming):
    def __init__(self,
                 inp, # HuggingFainpce Dataset or DatasetDict
                 main_text:str, # Name of the main text column
                 filter_dict={}, # A dictionary: {feature: filtering_function_for_that_feature}
                 metadatas=[], # Names of the metadata columns
                 process_metas=True, # Whether to do simple text processing on the chosen metadatas
                 content_transformations=[], # A list of text transformations
                 seed=None, # Random seed
                 batch_size=1024, # Transformation + Tokenization batch size
                 num_proc=1, # Number of process for multiprocessing
                 cols_to_keep=None, # Columns to keep after all processings
                 verbose=True, # Whether to prdint processing information
                ):
        
        super().__init__(inp=inp,
                         main_text=main_text,
                         filter_dict=filter_dict,
                         metadatas=metadatas,
                         process_metas=process_metas,
                         content_transformations=content_transformations,
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
    
    def _do_transformation_augmentation_tokenization(self):
        raise NotImplementedError("There's no augmentation in text processing for Language Model")


    def save_as_pickles(self,
                        fname, # Name of the pickle file
                        parent='pickle_files', # Parent folder
                       ):
        
        save_to_pickle(self,fname,parent=parent)
        
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
    
    
    def _do_transformation_tokenization(self,dtrain):             
        if len(self.content_tfms):            
            for tfm in self.content_tfms:
                _func = partial(lambda_map_batch,
                                feature=self.main_text,
                                func=tfm,
                                is_batched=self.is_batched)
                dtrain = hf_map_dset(dtrain,_func,self.is_batched,self.batch_size,self.num_proc)

        tok_func = partial(tokenize_function,
                           tok=self.tokenizer,
                           max_length=self.max_length if self.line_by_line else -1,
                           return_special_tokens_mask=True
                          )
        
        # Tokenization
        _func = partial(lambda_map_batch,
                        feature=self.main_text,
                        func=tok_func,
                        output_feature=None,
                        is_batched=self.is_batched)
        
        dtrain = hf_map_dset(dtrain,_func,self.is_batched,self.batch_size,self.tok_num_proc)
        if not self.line_by_line: dtrain = dtrain.remove_columns(self.cols_to_keep)   
        
        # Token concatenation
        if not self.line_by_line: 
            dtrain = hf_map_dset(dtrain,
                                 self._group_texts_with_stride,
                                 is_batched=True,
                                 batch_size=self.batch_size,
                                 num_proc=self.tok_num_proc)
        return dtrain
    
    
    def _construct_generator_with_batch(self,dset):        
        def _get_generator(dset):
            for v in dset: yield v
            
        final_dict = defaultdict(list)
        for inp in dset: # dset is generator
            # inp[text_name] will be a single item
            for k,v in inp.items():
                final_dict[k].append(v)
            
            if len(final_dict[self.main_text])==self.batch_size:
                # a full batch (self.batch_size) is created
                dtrain = Dataset.from_dict(final_dict)
                dtrain = self._do_transformation_tokenization(dtrain)
                yield from _get_generator(dtrain)
                final_dict=defaultdict(list)            
            
        if len(final_dict[self.main_text]):
            # hasn't reached batch_size (of last batch)
            dtrain = Dataset.from_dict(final_dict)
            dtrain = self._do_transformation_tokenization(dtrain)
            yield from _get_generator(dtrain)

    def _do_transformation_tokenization_generator(self):
        _tmp1 = self.num_proc
        _tmp2 = self.tok_num_proc
        self.num_proc=1
        self_tok_num_proc=1
        self.main_ddict['train'] = IterableDataset.from_generator(self._construct_generator_with_batch,
                                                                  gen_kwargs={'dset': self.main_ddict['train']}
                                                                 )
        self.num_proc = _tmp1
        self.tok_num_proc = _tmp2
    
    def _do_transformation_tokenization_generator_fast(self):
        # only use for line-by-line tokenization with no padding
        def _get_generator(dset,tok_func,all_tfms):
            for inp in dset:
                # inp[text_name] will be a single item
                results = tok_func(all_tfms(inp[self.main_text]))
                # add back cols_to_keep in inp
                results = dict(inp,**results)
                yield results
        
        # no padding for tokenization
        tok_func = partial(tokenize_function,
                           tok=self.tokenizer,
                           max_length=-1,
                           return_special_tokens_mask=True
                          )
        all_tfms = self.content_tfms 
        all_tfms = partial(func_all,functions=all_tfms) if len(all_tfms) else lambda x: x
           
        self.main_ddict['train'] = IterableDataset.from_generator(_get_generator,
                                                   gen_kwargs={'dset': self.main_ddict['train'],
                                                               'tok_func':tok_func,
                                                               'all_tfms': all_tfms
                                                              }
                                                                 )

    
    def process_and_tokenize(self,
                             tokenizer, # Tokenizer (preferably from HuggingFace)
                             max_length=None, # pad to model's allowed max length (default is max_sequence_length). Use -1 for no padding at all
                             tok_num_proc=None, # Number of processes for tokenization
                             line_by_line=True, # To whether tokenize each sentence separately, or concatenate them
                             stride=None, # option to do striding when line_by_line is False
                            ):
        if self._processed_call:
            warnings.warn('Your dataset has already been processed. Returning the previous processed DatasetDict...')
            return self.main_ddict
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_by_line = line_by_line
        if not self.line_by_line and self.batch_size==1:
            raise ValueError('Cannot perform token concatenation with batch size of 1')
        self.stride = stride        
        self.tok_num_proc = tok_num_proc if tok_num_proc else self.num_proc
        
        # Filtering
        print_msg('Data Filtering',20,verbose=self.verbose)
        for k in self.main_ddict.keys():   
            self.main_ddict[k] = self._do_filtering(self.main_ddict[k])
        self.verboseprint('Done')

        
        # Process metadatas
        print_msg('Metadata Simple Processing & Concatenating to Main Content',verbose=self.verbose)
        for k in self.main_ddict.keys():   
            self.main_ddict[k] = self._process_metadatas(self.main_ddict[k])
        self.verboseprint('Done')

        # Dropping unused columns
        self._simplify_ddict()

        if self.seed:
            seed_everything(self.seed)
            
        # Content transformation + tokenization for validation
        if 'validation' in self.main_ddict.keys():
            print_msg('Performing Content Transformation and Tokenization on Validation Set',verbose=self.verbose)
            self.main_ddict['validation'] = self._do_transformation_tokenization(self.main_ddict['validation'])
            self.verboseprint('Done')
        
        # Content transformation + tokenization for train
        print_msg('Creating a generator for content transformation and tokenization on Train set',verbose=self.verbose)
        if line_by_line and max_length is not None and max_length<0: # line-by-line tokenization with no padding
            self._do_transformation_tokenization_generator_fast()
        else:
            self._do_transformation_tokenization_generator()
        self.verboseprint('Done')
        
        self._processed_call=True
    
    def set_data_collator(self,
                          is_mlm=True, # Is this masked language model (True) or causal language model (False)
                          mlm_prob=0.15, # Mask probability for masked language model
                         ):
        if not hasattr(self,'max_length'):
            raise ValueError("Please call `process_and_tokenize' or `do_tokenization` to tokenize your dataset")
            
        pad_to_multiple_of_8 = (self.max_length<0) # get data collator to pad when tokenizer does not apply padding
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
        
        # Drop unused columns
        cols_to_remove = test_cols - set(self.cols_to_keep)
        test_dset = test_dset.remove_columns(list(cols_to_remove))
        
        if do_tokenize:
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

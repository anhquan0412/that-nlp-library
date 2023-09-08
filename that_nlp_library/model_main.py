# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_model_main.ipynb.

# %% ../nbs/03_model_main.ipynb 3
from __future__ import annotations
import os, sys
from transformers import Trainer, TrainingArguments, AutoConfig
from datasets import DatasetDict,Dataset
import torch
import gc
from sklearn.metrics import accuracy_score
from functools import partial
import numpy as np
from .utils import *
from .text_main import TextDataController
from .text_main_streaming import TextDataControllerStreaming

# %% auto 0
__all__ = ['model_init_classification', 'compute_metrics', 'compute_metrics_separate_heads', 'loss_for_classification',
           'finetune', 'ModelController']

# %% ../nbs/03_model_main.ipynb 4
def model_init_classification(
                              model_class, # Model's class object, e.g. RobertaHiddenStateConcatForSequenceClassification
                              cpoint_path, # Either model string name on HuggingFace, or the path to model checkpoint
                              output_hidden_states:bool, # To whether output the model hidden states or not. Useful when you try to build a custom classification head 
                              device=None, # Device to train on
                              config=None, # Model config. If not provided, AutoConfig is used to load config from cpoint_path
                              seed=None, # Random seed
                              body_model=None, # If not none, we use this to initialize model's body. If you only want to load the model checkpoint in cpoint_path, leave this as none
                              model_kwargs={} # Keyword arguments for model (both head and body)
                             ):
    """To initialize a classification (or regression) model, either from an existing HuggingFace model or custom architecture
    
    Can be used for binary, multi-class single-head, multi-class multi-head, multi-label clasisifcation, and regression
    """
    if device is None: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config is None:
        config = AutoConfig.from_pretrained(
            cpoint_path,
            output_hidden_states=output_hidden_states,
        )
    else:
        config.output_hidden_states=output_hidden_states
    
    
    seed_everything(seed)
    if body_model is not None:
        model = model_class(config=config,**model_kwargs)
        layers = list(model.children())
        print('Loading body weights. This assumes the body is the very first block of your custom architecture')
        body_name, _ = next(iter(model.named_children()))
        setattr(model, body_name, body_model)
        model = model.to(device)
        
    else:
        model = model_class.from_pretrained(cpoint_path,config=config,**model_kwargs).to(device)
    return model

# %% ../nbs/03_model_main.ipynb 7
def compute_metrics(pred, # An EvalPrediction object from HuggingFace (which is a named tuple with ```predictions``` and ```label_ids``` attributes)
                    metric_funcs=[], # A list of metric functions to evaluate
                    head_sizes=[], # Class size for each head. Regression head will have head size 1
                    label_names=[], # Names of the label (dependent variable) columns
                    is_multilabel=False, # Whether this is a multilabel classification
                    multilabel_threshold=0.5 # Threshold for multilabel (>= threshold is positive)
                   ):
    """
    Return a dictionary of metric name and its values.
    
    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L107C16-L107C16
    """
    assert len(head_sizes)==len(label_names)
    labels = pred.label_ids 
    if isinstance(pred.predictions,tuple):
        preds = pred.predictions[0]
    else:
        preds = pred.predictions
        
    results={}
    metric_funcs = val2iterable(metric_funcs)
    
    for i,(_size,_name) in enumerate(zip(head_sizes,label_names)):
        start= 0 if i==0 else start+head_sizes[i-1]
        end = start + _size
        _pred = preds[:,start:end]
        if is_multilabel:
            # sigmoid and threshold
            _pred = (sigmoid(_pred)>=multilabel_threshold).astype(int)
        elif _size>1: # classification
            _pred = _pred.argmax(-1)
        _label = labels[:,i] if len(head_sizes)>1 else labels
        for m_func in metric_funcs:
            m_name = callable_name(m_func)
            results[f'{m_name}_{_name}']=m_func(_label,_pred)
    return results

# %% ../nbs/03_model_main.ipynb 9
def compute_metrics_separate_heads(pred, # An EvalPrediction object from HuggingFace (which is a named tuple with ```predictions``` and ```label_ids``` attributes)
                              metric_funcs=[], # A list of metric functions to evaluate
                              label_names=[], # Names of the label (dependent variable) columns
                              **kwargs
                             ):
    """
    Return a dictionary of metric name and its values. This is used in Deep Hierarchical Classification (special case of multi-head classification)
    
    This metric function is mainly used when you have a separate logit output for each head 
    (instead of the typical multi-head logit output: all heads' logits are concatenated)
    """
    # pred: EvalPrediction object 
    # (which is a named tuple with predictions and label_ids attributes)
    labels = pred.label_ids # (bs,number of head separately)
    assert labels.shape[1]==len(label_names)
    
    results={}
    metric_funcs = val2iterable(metric_funcs)
    
    for i in range(len(label_names)):
        _label = labels[:,i]
        _pred = pred.predictions[i].argmax(-1)
        for m_func in metric_funcs:
            m_name = callable_name(m_func)
            results[f'{m_name}_{label_names[i]}']=m_func(_label,_pred)
    
    return results

# %% ../nbs/03_model_main.ipynb 13
def loss_for_classification(logits, # output of the last linear layer, before any softmax/sigmoid. Size: (bs,class_size)
                            labels, # determined by your datasetdict. Size: (bs,number_of_head)
                            is_multilabel=False, # Whether this is a multilabel classification
                            is_multihead=False, # Whether this is a multihead classification
                            head_sizes=[], # Class size for each head. Regression head will have head size 1
                            head_weights=[], # loss weight for each head. Default to 1 for each head
                           ):
    """
    The general loss function for classification
    
    - If is_multilabel is ```False``` and is_multihead is ```False```: Single-Head Classification, e.g. You predict 1 out of n class
    
    - If is_multilabel is ```False``` and is_multihead is ```True```: Multi-Head Classification, e.g. You predict 1 out of n classes at Level 1, 
    and 1 out of m classes at Level 2
    
    - If is_multilabel is ```True``` and is_multihead is ```False```: Single-Head Multi-Label Classification, e.g. You predict x out of n class (x>=0)
    
    - If is_multilabel is ```True``` and is_multihead is ```True```: Not supported
    
    """
    if is_multilabel and is_multihead: raise ValueError('Multi-Label and Multi-Head problem is not supported')
    head_sizes = val2iterable(head_sizes)
    if len(head_sizes) and not len(head_weights):
        head_weights = val2iterable(1,len(head_sizes))
        
    loss=0
    if not is_multilabel:
        assert len(head_sizes)==len(head_weights),"Make sure len of head_sizes and len of head_weights are equal"
        for i,(_size,_weight) in enumerate(zip(head_sizes,head_weights)):
            start= 0 if i==0 else start+head_sizes[i-1]
            end = start + _size
            
            loss_fct = torch.nn.MSELoss() if _size==1 else torch.nn.CrossEntropyLoss()
            
            _logits = logits[:,start:end] # (bs, _size)
            _logits = _logits.squeeze() if _size ==1 else _logits.view(-1,_size)
            
            _label = labels[:,i] if len(head_sizes)>1 else labels # (bs,) for 1 total 1 head size, (bs,num_head) otherwise
            
            loss = loss + _weight*loss_fct(_logits,_label.view(-1))
    else:
        if not is_multihead:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits,
                            labels.float())
                    
    return loss

# %% ../nbs/03_model_main.ipynb 15
def finetune(lr, # Learning rate
             bs, # Batch size
             wd, # Weight decay
             epochs, # Number of epochs
             ddict, # The HuggingFace datasetdict
             tokenizer,# HuggingFace tokenizer
             o_dir = './tmp_weights', # Directory to save weights
             save_checkpoint=False, # Whether to save weights (checkpoints) to o_dir
             model=None, # NLP model
             model_init=None, # A function to initialize model
             data_collator=None, # HuggingFace data collator
             compute_metrics=None, # A function to compute metric, e.g. `compute_metrics`
             grad_accum_steps=2, # The batch at each step will be divided by this integer and gradient will be accumulated over gradient_accumulation_steps steps.
             lr_scheduler_type='cosine',  # The scheduler type to use. Including: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
             warmup_ratio=0.1, # The warmup ratio for some lr scheduler
             no_valid=False, # Whether there is a validation set or not
             seed=None, # Random seed
             report_to='none', # The list of integrations to report the results and logs to. Supported platforms are "azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard","clearml" and "wandb". Use "all" to report to all integrations installed, "none" for no integrations.
             trainer_class=None, # You can include the class name of your custom trainer here
            ):
    "The main model training/finetuning function"
    torch.cuda.empty_cache()
    gc.collect()

    seed_everything(seed)
    training_args = TrainingArguments(o_dir, 
                                learning_rate=lr, 
                                warmup_ratio=warmup_ratio,
                                lr_scheduler_type=lr_scheduler_type, 
                                fp16=True,
                                do_train=True,
                                do_eval= not no_valid,
                                evaluation_strategy="no" if no_valid else "epoch", 
                                save_strategy="epoch" if save_checkpoint else 'no',
                                overwrite_output_dir=True,
                                gradient_accumulation_steps=grad_accum_steps,
                                per_device_train_batch_size=bs, 
                                per_device_eval_batch_size=bs,
                                num_train_epochs=epochs, weight_decay=wd,
                                report_to=report_to,
                                logging_dir=os.path.join(o_dir, 'log') if report_to!='none' else None,
                                logging_steps = len(ddict["train"]) // bs,
                                )

    # instantiate trainer
    trainer_class = Trainer if trainer_class is None else trainer_class
    trainer = trainer_class(
        model=model,
        model_init=model_init if model is None else None,
        args=training_args,
        train_dataset=ddict['train'],#.shard(200, 0)
        eval_dataset=ddict['validation'] if not no_valid else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    
    trainer.train()
    return trainer

# %% ../nbs/03_model_main.ipynb 19
def _forward_pass_prediction(batch,
                             model=None, # NLP model
                             topk=1, # Number of labels to return for each head
                             is_multilabel=False, # Is this a multilabel classification?
                             multilabel_threshold=0.5, # The threshold for multilabel classification
                             model_input_names=['input_ids', 'token_type_ids', 'attention_mask'], # Model required inputs, from tokenizer.model_input_names
                             data_collator=None, # HuggingFace data collator
                             label_names=[], # Names of the label columns
                             head_sizes=[], # Class size for each head. Regression head will have head size 1
                             device = None, # device that the model is trained on
                             are_heads_separated=False, # is this multi-head, but each head has a separated logit?
                             ):
    if data_collator is not None:
        
# --- Convert from  
# {'input_ids': [tensor([    0, 10444,   244, 14585,   125,  2948,  5925,   368,     2]), 
#                tensor([    0, 16098,  2913,   244,   135,   198, 34629,  6356,     2])]
# 'attention_mask': [tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 
#                    tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])]
#                    }
# --- to
# [{'input_ids': tensor([    0, 10444,   244, 14585,   125,  2948,  5925,   368,     2]),
#   'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])},
#  {'input_ids': tensor([    0, 16098,  2913,   244,   135,   198, 34629,  6356,     2]),
#   'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])}]

        # remove string text, due to transformer new version       
        collator_inp = []
        ks = [k for k in batch.keys() if k in model_input_names+['label']]
        vs = [batch[k] for k in ks]
        for pair in zip(*vs):
            collator_inp.append({k:v for k,v in zip(ks,pair)})
        
        batch = data_collator(collator_inp)
    
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in model_input_names}
    
    assert len(label_names)==len(head_sizes), "Length of `label_names` must equal to length of `head_sizes`"
        
    # switch to eval mode for evaluation
    if model.training:
        model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        outputs_logits = outputs.logits
        results={}
        for i in range(len(label_names)):
            if are_heads_separated:
                _f = partial(torch.nn.functional.softmax,dim=1)
                _preds = _f(outputs_logits[i].cpu())
            else:
                outputs_logits = outputs_logits.cpu()
                start = 0 if i==0 else start+head_sizes[i-1]
                end = start + head_sizes[i]
                if head_sizes[i]==1: # regression
                    _f = lambda x: x.squeeze()
                else: # multilabel, or classification
                    _f = partial(torch.nn.functional.softmax,dim=1) if not is_multilabel else torch.sigmoid 
                _preds = _f(outputs_logits[:,start:end])
        
            if is_multilabel:
                _pred_labels = _preds>=multilabel_threshold
                results[f'pred_{label_names[i]}']=_pred_labels.numpy()
                results[f'pred_prob_{label_names[i]}']=_preds.numpy()
            else:
                if head_sizes[i]==1: # regression
                    results[f'pred_{label_names[i]}']=_preds.numpy()
                else: # classification
                    _p,_l = torch.topk(_preds,topk,dim=-1)
                    if topk==1:
                        _l,_p = _l[:,0],_p[:,0]
                    results[f'pred_{label_names[i]}']=_l.numpy()
                    results[f'pred_prob_{label_names[i]}']=_p.numpy()    

    
    # Switch back to train mode
    if not model.training:
        model.train()
    
    return results

# %% ../nbs/03_model_main.ipynb 20
def _convert_pred_id_to_label(dset,
                              label_names,
                              label_lists,
                              topk=1,
                              is_multilabel=False,
                              batch_size=1000,
                              num_proc=1
                             ):
    is_batched=batch_size>1
    if is_multilabel:
        get_label_str_multilabel = lambda x: [label_lists[0][int(j)] for j in np.where(x==True)[0]]
        _func = partial(lambda_map_batch,feature=f'pred_{label_names[0]}',
                        func=get_label_str_multilabel,
                        is_batched=is_batched
                       )
        dset = hf_map_dset(dset,_func,
                           is_batched=is_batched,
                           batch_size=batch_size,
                           num_proc=num_proc
                          )
        return dset
    
    for i in range(len(label_names)):
        if len(label_lists[i])==0: # regression
            _func1 = lambda xs: xs
        else: # classification
            _func1 = lambda xs: label_lists[i][int(xs)] if not isinstance(xs,np.ndarray) else [label_lists[i][int(x)] for x in xs]
        
        _func2 = partial(lambda_map_batch,
                         feature=f'pred_{label_names[i]}',
                        func=_func1,
                        is_batched=is_batched
                       )
        dset = hf_map_dset(dset,_func2,
                           is_batched=is_batched,
                           batch_size=batch_size,
                           num_proc=num_proc
                           )
    return dset


# %% ../nbs/03_model_main.ipynb 21
class ModelController():
    def __init__(self,
                 model, # NLP model
                 data_store=None, # a TextDataController/TextDataControllerStreaming object
                 seed=None, # Random seed
                ):
        self.model = model
        self.data_store = data_store
        self.seed = seed
        
    def fit(self,
            epochs, # Number of epochs
            learning_rate, # Learning rate
            ddict=None, # DatasetDict to fit (will override data_store)
            metric_funcs=[accuracy_score], # Metric function (can be from Sklearn)
            batch_size=16, # Batch size
            weight_decay=0.01, # Weight decay
            lr_scheduler_type='cosine', # The scheduler type to use. Including: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
            warmup_ratio=0.1, # The warmup ratio for some lr scheduler
            o_dir = './tmp_weights', # Directory to save weights
            save_checkpoint=False, # Whether to save weights (checkpoints) to o_dir
            hf_report_to='none', # The list of HuggingFace-allowed integrations to report the results and logs to
            compute_metrics=None, # A function to compute metric, e.g. `compute_metrics` which utilizes the given ```metric_funcs``` 
            grad_accum_steps=2, # Gradient will be accumulated over gradient_accumulation_steps steps.
            tokenizer=None, # Tokenizer (to override one in ```data_store```)
            data_collator=None, # Data Collator (to override one in ```data_store```)
            label_names=None, # Names of the label (dependent variable) columns (to override one in ```data_store```)
            head_sizes=None, # Class size for each head (to override one in ```model```)
            trainer_class=None, # You can include the class name of your custom trainer here
           ):
        
        if tokenizer is None: tokenizer=check_and_get_attribute(self.data_store,'tokenizer')
        if data_collator is None: data_collator=getattr(self.data_store,'data_collator',None)
        if ddict is None: ddict = check_and_get_attribute(self.data_store,'main_ddict')
            
        if label_names is None: label_names=check_and_get_attribute(self.data_store,'label_names')
        label_names = val2iterable(label_names)
        
        if head_sizes is None: 
            head_sizes=check_and_get_attribute(self.data_store,'label_lists')
            head_sizes = [len(hs) if len(hs) else 1 for hs in head_sizes]
        head_sizes = val2iterable(head_sizes)
        
        if len(set(ddict.keys()) & set(['train','training']))==0:
            raise ValueError("Missing the following key for DatasetDict: train/training")
        no_valid = len(set(ddict.keys()) & set(['validation','val','valid']))==0

        _compute_metrics = partial(compute_metrics,
                                   metric_funcs=metric_funcs,
                                   head_sizes=head_sizes,
                                   label_names=label_names 
                                  )
        
        trainer = finetune(learning_rate,batch_size,weight_decay,epochs,
                           ddict,tokenizer,o_dir,
                           save_checkpoint=save_checkpoint,
                           model=self.model,
                           data_collator=data_collator,
                           compute_metrics=_compute_metrics,
                           grad_accum_steps=grad_accum_steps,
                           lr_scheduler_type=lr_scheduler_type,
                           warmup_ratio=warmup_ratio,
                           no_valid=no_valid,
                           seed=self.seed,
                           trainer_class=trainer_class,
                           report_to=hf_report_to)
        self.trainer = trainer
        
    def predict_raw_text(self,
                         content:dict|list|str, # Either a single sentence, list of sentence or a dictionary where keys are metadata, values are list
                         batch_size=1, # Batch size. For a small amount of texts, you might want to keep this small
                         is_multilabel=None, # Is this a multilabel classification?
                         multilabel_threshold=0.5, # Threshold for multilabel classification
                         topk=1, # Number of labels to return for each head
                         are_heads_separated=False # Are outpuf (of model) separate heads?
                        ):
        if not isinstance(self.data_store,(TextDataController,TextDataControllerStreaming)) or not self.data_store._processed_call:
            raise ValueError('This functionality needs a TextDataController object which has processed some training data')
#         with HiddenPrints():
        test_dset = self.data_store.prepare_test_dataset_from_raws(content)

        results = self.predict_ddict(ddict=test_dset,
                                     batch_size=batch_size,
                                     is_multilabel=is_multilabel,
                                     multilabel_threshold=multilabel_threshold,
                                     topk=topk,
                                     are_heads_separated=are_heads_separated
                                    )
        return results.to_pandas()
    
    def predict_raw_dset(self,
                         dset, # A raw HuggingFace dataset
                         batch_size=16, # Batch size. For a small amount of texts, you might want to keep this small
                         do_filtering=False, # Whether to perform data filtering on this test set
                         is_multilabel=None, # Is this a multilabel classification?
                         multilabel_threshold=0.5, # Threshold for multilabel classification
                         topk=1, # Number of labels to return for each head
                         are_heads_separated=False # Are outpuf (of model) separate heads?
                        ):
        if not isinstance(self.data_store,(TextDataController,TextDataControllerStreaming)) or not self.data_store._processed_call:
            raise ValueError('This functionality needs a TextDataController object which has processed some training data')
#         with HiddenPrints():
        test_dset = self.data_store.prepare_test_dataset(dset,do_filtering)

        results = self.predict_ddict(test_dset,
                                     batch_size=batch_size,
                                     is_multilabel=is_multilabel,
                                     multilabel_threshold=multilabel_threshold,
                                     topk=topk,
                                     are_heads_separated=are_heads_separated
                                    )
        return results
    
    def predict_ddict(self,
                      ddict:DatasetDict|Dataset=None, # A processed and tokenized DatasetDict/Dataset (will override one in ```data_store```)
                      ds_type='test', # The split of DatasetDict to predict
                      batch_size=16, # Batch size for making prediction on GPU
                      is_multilabel=None, # Is this a multilabel classification?
                      multilabel_threshold=0.5, # Threshold for multilabel classification
                      topk=1, # Number of labels to return for each head
                      tokenizer=None, # Tokenizer (to override one in ```data_store```)
                      data_collator=None, # Data Collator (to override one in ```data_store```)
                      label_names=None, # Names of the label (dependent variable) columns (to override one in ```data_store```)
                      class_names_predefined=None, # List of names associated with the labels (same index order) (to override one in ```data_store```)
                      device=None, # Device that the model is trained on
                      are_heads_separated=False # Are outputs (of model) separate heads?
                     ):
        if device is None: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_multilabel is None: is_multilabel=getattr(self.model,'is_multilabel',False)
        label_lists = class_names_predefined
        if tokenizer is None: tokenizer=check_and_get_attribute(self.data_store,'tokenizer')
        if data_collator is None: data_collator=getattr(self.data_store,'data_collator',None)
        if label_names is None: label_names=check_and_get_attribute(self.data_store,'label_names')
        if label_lists is None: label_lists = check_and_get_attribute(self.data_store,'label_lists')
        label_names = val2iterable(label_names)
        if not isinstance(label_lists[0],list):
            label_lists=[label_lists]    
        head_sizes = [len(hs) if len(hs) else 1 for hs in label_lists]
        if ddict is None: ddict = check_and_get_attribute(self.data_store,'main_ddict')
        if isinstance(ddict,DatasetDict):
            if ds_type not in ddict.keys():
                raise ValueError(f'{ds_type} is not in the given DatasetDict')
            ddict = ddict[ds_type]
            
        ddict.set_format("torch",
                        columns=tokenizer.model_input_names)
        
        print_msg('Start making predictions',20)
        # this will create features: pred_classname and/or pred_prob_classname
        results = ddict.map(
                            partial(_forward_pass_prediction,
                                    model=self.model,
                                    topk=topk,
                                    is_multilabel=is_multilabel,
                                    multilabel_threshold=multilabel_threshold,
                                    model_input_names=tokenizer.model_input_names,
                                    data_collator=data_collator,
                                    label_names=label_names,
                                    head_sizes=head_sizes,
                                    are_heads_separated = are_heads_separated,
                                    device=device
                                   ), 
                            batched=True, 
                            batch_size=batch_size)
        
        # convert all to numpy
        results.set_format('numpy')
            
        results = _convert_pred_id_to_label(results,
                                            label_names,
                                            label_lists,
                                            topk,
                                            is_multilabel,
                                            batch_size=1000,
                                            num_proc=1
                                           )
        return results
        

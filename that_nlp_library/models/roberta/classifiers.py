# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/04_models.roberta.classifiers.ipynb.

# %% ../../../nbs/04_models.roberta.classifiers.ipynb 3
from __future__ import annotations
import torch
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from ...model_main import loss_for_classification
from ...utils import *

# %% auto 0
__all__ = ['ConcatHeadExtended', 'ConcatHeadSimple', 'RobertaClassificationHeadCustom', 'RobertaBaseForSequenceClassification',
           'RobertaHiddenStateConcatForSequenceClassification']

# %% ../../../nbs/04_models.roberta.classifiers.ipynb 5
class ConcatHeadExtended(torch.nn.Module):
    """
    Concatenated head for Roberta Classification Model. 
    This head takes the last n hidden states of [CLS], and concatenate them before passing through the classifier head
    """
    def __init__(self,
                 config, # HuggingFace model configuration
                 classifier_dropout=0.1, # Dropout ratio (for dropout layer right before the last nn.Linear)
                 last_hidden_size=768, # Last hidden size (before the last nn.Linear)
                 layer2concat=4, # number of hidden layer to concatenate (counting from top)
                 num_labels=None, # Number of label output. Overwrite config.num_labels 
                 **kwargs
                ):

        super().__init__()
        self.last_hidden_size=last_hidden_size
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.pre_classifier = torch.nn.Linear(layer2concat*config.hidden_size,last_hidden_size)
        num_labels=num_labels if num_labels is not None else config.num_labels
        self.out_proj = torch.nn.Linear(last_hidden_size, num_labels)
    
    def forward(self, inp, **kwargs):
        x = inp
        x = self.dropout(x)
        x = self.pre_classifier(x)
        x = torch.tanh(x)
#         x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# %% ../../../nbs/04_models.roberta.classifiers.ipynb 7
class ConcatHeadSimple(torch.nn.Module):
    """
    Concatenated head for Roberta Classification Model, the simpler version (no hidden linear layer)
    This head takes the last n hidden states of [CLS], and concatenate them before passing through the classifier head
    """
    def __init__(self,
                 config, # HuggingFace model configuration
                 classifier_dropout=0.1, # Dropout ratio (for dropout layer right before the last nn.Linear)
                 layer2concat=4, # number of hidden layer to concatenate (counting from top)
                 num_labels=None, # Number of label output. Overwrite config.num_labels 
                 **kwargs
                ):

        super().__init__()
        self.dropout = torch.nn.Dropout(classifier_dropout)
        num_labels=num_labels if num_labels is not None else config.num_labels
        self.out_proj = torch.nn.Linear(layer2concat*config.hidden_size, num_labels)
    def forward(self, inp, **kwargs):
        x = inp
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# %% ../../../nbs/04_models.roberta.classifiers.ipynb 9
class RobertaClassificationHeadCustom(torch.nn.Module):
    """
    Same as RobertaClassificationHead, but you can freely adjust dropout
    
    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L1424
    """
    
    def __init__(self, 
                 config, # HuggingFace model configuration
                 classifier_dropout=0.1, # Dropout ratio (for dropout layer right before the last nn.Linear)
                 num_labels=None, # Number of label output. Overwrite config.num_labels 
                 **kwargs
                ):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(classifier_dropout)
        num_labels=num_labels if num_labels is not None else config.num_labels
        self.out_proj = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, inp, **kwargs):
        x = self.dropout(inp)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# %% ../../../nbs/04_models.roberta.classifiers.ipynb 12
class RobertaBaseForSequenceClassification(RobertaPreTrainedModel):
    """
    Base Roberta Architecture for Sequence Classification task
    
    Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L1155C35-L1155C35
    """
    # make sure standard XLM-R are used
    config_class = RobertaConfig

    def __init__(self,
                 config, # HuggingFace model configuration
                 is_multilabel=False, # Whether this is a multilabel classification
                 is_multihead=False, # Whether this is a multihead (multi-level) classification
                 head_class_sizes=[], # Class size for each head
                 head_weights=[], # loss weight for each head. This will be multiplied to the loss of each head's output
                 head_class=None, # The class object of the head. You can use RobertaClassificationHeadCustom as default
                 **head_class_kwargs, # Keyword arguments for the head class
                ):
        super().__init__(config)
        self.is_multilabel = is_multilabel
        self.is_multihead = is_multihead
        self.head_class_sizes = val2iterable(head_class_sizes)
        self.head_weights = val2iterable(head_weights,lsize=len(self.head_class_sizes))
        
        # set num_labels for config
        num_labels = sum(self.head_class_sizes)
        config.num_labels = num_labels
        
        # add_pooling_layer to False to ensure all hidden states are returned and not only the one associated with the [CLS] token.
        self.body_model = RobertaModel(config, add_pooling_layer=False)
        # Set up head
        self.head = head_class(config=config,**head_class_kwargs)


    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        token_type_ids= None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        labels= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
        ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs = self.body_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        logits = self.head(sequence_output) # (bs,sum of all class sizes)
        
        # Calculate losses
        if labels is None:
            loss=None
        else:
            loss = loss_for_classification(logits, labels, 
                                   self.is_multilabel,
                                   self.is_multihead, 
                                   self.head_class_sizes,
                                   self.head_weights)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits,
#                                      hidden_states=outputs.hidden_states,
                                        hidden_states=None,
                                        attentions=outputs.attentions)

# %% ../../../nbs/04_models.roberta.classifiers.ipynb 14
class RobertaHiddenStateConcatForSequenceClassification(RobertaPreTrainedModel):
    """
    Roberta Architecture with Hidden-State-Concatenation for Sequence Classification task
    """
    
    config_class = RobertaConfig

    def __init__(self,config, # HuggingFace model configuration
                 layer2concat=4, # number of hidden layer to concatenate (counting from top)
                 is_multilabel=False, # Whether this is a multilabel classification
                 is_multihead=False, # Whether this is a multihead (multi-level) classification
                 head_class_sizes=[], # Class size for each head
                 head_weights=[], # loss weight for each head. This will be multiplied to the loss of each head's output
                 head_class=None, # The class object of the head. You can use ConcatHeadSimple or ConcatHeadExtended
                 **head_class_kwargs, # Keyword arguments for the head class
                ):
        super().__init__(config)
        self.is_multilabel = is_multilabel
        self.is_multihead = is_multihead
        self.head_class_sizes = val2iterable(head_class_sizes)
        self.head_weights = val2iterable(head_weights,lsize=len(self.head_class_sizes))
        self.layer2concat=layer2concat
        # set num_labels for config
        num_labels = sum(self.head_class_sizes)
        config.num_labels = num_labels
        
        # Load model body
        # add_pooling_layer to False to ensure all hidden states are returned  and not only the one associated with the [CLS] token.
        self.body_model = RobertaModel(config, add_pooling_layer=False)
        
        # Set up head
        self.head = head_class(config=config,layer2concat=layer2concat,
                                              **head_class_kwargs)

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        token_type_ids= None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        labels= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
        ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs = self.body_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs['hidden_states'] # tuples with len 13 (number of layer/block)
        # each with shape: (bs,seq_len,hidden_size_len), e.g. for phobert: (bs,256, 768)
        # Note: hidden_size_len = embedding_size
        hidden_concat = torch.cat([hidden_states[i][:,0] for i in range(-1,-self.layer2concat-1,-1)],
                                  -1) 
        logits = self.head(hidden_concat) # (bs,sum of all class sizes)
        
        # Calculate losses
        if labels is None:
            loss=None
        else:            
            loss = loss_for_classification(logits, labels, 
                                   self.is_multilabel,
                                   self.is_multihead, 
                                   self.head_class_sizes,
                                   self.head_weights)
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss, logits=logits,
#                                      hidden_states=outputs.hidden_states,
                                        hidden_states=None,
                                        attentions=outputs.attentions)

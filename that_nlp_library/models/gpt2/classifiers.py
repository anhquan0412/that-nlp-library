# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/09_models.gpt2.classifiers.ipynb.

# %% ../../../nbs/09_models.gpt2.classifiers.ipynb 3
from __future__ import annotations
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Model,GPT2PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from ...model_main import loss_for_classification
from ...utils import *

# %% auto 0
__all__ = ['GPT2BaseForSequenceClassification', 'GPT2HiddenStateConcatForSequenceClassification']

# %% ../../../nbs/09_models.gpt2.classifiers.ipynb 5
class GPT2BaseForSequenceClassification(GPT2PreTrainedModel):
    """
    GPT2 Architecture for Sequence Classification task
    Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1376
    """
    config_class = GPT2Config

    def __init__(self,
                 config, # HuggingFace model configuration
                 is_multilabel=False, # Whether this is a multilabel classification
                 is_multihead=False, # Whether this is a multihead (multi-level) classification
                 head_class_sizes=[], # Class size for each head
                 head_weights=[], # loss weight for each head. This will be multiplied to the loss of each head's output
                 head_class=None, # The class object of the head. 
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
        
        self.body_model = GPT2Model(config)
        
        # Set up token classification head
        if head_class is None:
            self.head = torch.nn.Linear(config.n_embd, num_labels, bias=False)
        else:
            self.head = head_class(config,**head_class_kwargs)

        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        self.config.pad_token_id = self.config.eos_token_id
        
    def forward(
        self,
        input_ids= None,
        past_key_values= None,
        attention_mask= None,
        token_type_ids= None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
    ):
        # Use model body to get encoder representations
        # the only ones we need for now are input_ids and attention_mask
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs = self.body_model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
              
        sequence_output = outputs[0] # last hidden state: (bs,sequence_length,hidden_size: 768)
        
        # get the idx of the last token (typically just -1), to be used for classification
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    sequence_output.device
                )
            else:
                sequence_lengths = -1
                print(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
                
        # sequence length at this point is just the idx (or indices) of the last token        
        sequence_output = sequence_output[torch.arange(batch_size, device=sequence_output.device), 
                                          sequence_lengths,:] # (bs,hidden_sizes)
        
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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # Return model output object
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=outputs.attentions,
        )

# %% ../../../nbs/09_models.gpt2.classifiers.ipynb 7
class GPT2HiddenStateConcatForSequenceClassification(GPT2PreTrainedModel):
    """
    GPT2 Architecture for Sequence Classification task
    Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1376
    """
    config_class = GPT2Config

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
        
        self.body_model = GPT2Model(config)
        
        # Set up classification head
        self.head = head_class(config=config,layer2concat=layer2concat,
                                              **head_class_kwargs)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.config.pad_token_id = self.config.eos_token_id

    def forward(
        self,
        input_ids= None,
        past_key_values= None,
        attention_mask= None,
        token_type_ids= None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None):
        
        
        # Use model body to get encoder representations
        # the only ones we need for now are input_ids and attention_mask
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        outputs = self.body_model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
              
        sequence_output = outputs[0] # (bs,sequence_length,hidden_size)
        hidden_states = outputs['hidden_states'] # tuples with 12 layers
        
        # get the idx of the last token (typically just -1), to be used for classification
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
#                 sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                                    sequence_output.device
                                )
            else:
                sequence_lengths = -1
                print(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        # sequence length at this point is just the idx (or indices) of the last token        
        
        hidden_concat = torch.cat([hidden_states[i][torch.arange(batch_size, device=sequence_output.device), sequence_lengths,:] for i in range(-1,-self.layer2concat-1,-1)],
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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # Return model output object
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=outputs.attentions,
        )
    

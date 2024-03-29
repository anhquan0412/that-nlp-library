# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/06_models.roberta.conditional_prob_classifiers.ipynb.

# %% ../../../nbs/06_models.roberta.conditional_prob_classifiers.ipynb 3
from __future__ import annotations
import numpy as np
import torch
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaModel
from sklearn.preprocessing import MultiLabelBinarizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

# %% auto 0
__all__ = ['build_standard_condition_mask', 'RobertaHSCCProbSequenceClassification']

# %% ../../../nbs/06_models.roberta.conditional_prob_classifiers.ipynb 5
def build_standard_condition_mask(df_labels,
                                  label1,label2):
    L1_SIZE = df_labels[label1].nunique()
    L2_SIZE = df_labels[label2].nunique()

    
    df_labels = df_labels.drop_duplicates().sort_values([label1,label2])
    _d = df_labels.groupby([label1])[label2].apply(list).to_dict()
    
    mask_l1 = torch.eye(L1_SIZE) ==1
    mlb = MultiLabelBinarizer()
    mlb.fit([np.arange(L2_SIZE)])
    mask_l2 = torch.tensor(mlb.transform(list(_d.values())) == 1)
    
    mask_final= torch.cat((mask_l1,mask_l2),1)
    
    return mask_final

# %% ../../../nbs/06_models.roberta.conditional_prob_classifiers.ipynb 8
class RobertaHSCCProbSequenceClassification(RobertaPreTrainedModel):
    """
    Roberta Conditional Probability Architecture with Hidden-State-Concatenation for Sequence Classification task
    """
    config_class = RobertaConfig

    def __init__(self, 
                 config, # HuggingFace model configuration
                 size_l1=None, # Number of classes for head 1
                 size_l2=None, # Number of classes for head 2
                 standard_mask=None, # Mask for conditional probability
                 layer2concat=4, # number of hidden layer to concatenate (counting from top)
                 device=None, # CPU or GPU
                 head_class=None, # The class object of the head. You can use RobertaClassificationHeadCustom as default
                 **head_class_kwargs, # Keyword arguments for the head class
                ):
        super().__init__(config)
        self.training_device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size_l1 = size_l1
        self.size_l2 = size_l2
        self.layer2concat=layer2concat
        self.head_class_sizes=[size_l1,size_l2] # will be useful for metric calculation later
        
        # set num_labels for config
        num_labels = size_l1+size_l2
        config.num_labels = num_labels
        
        self.body_model = RobertaModel(config, add_pooling_layer=False)
        self.standard_mask = standard_mask.to(self.training_device)
        self.classification_head = head_class(config=config,layer2concat=layer2concat,
                                              **head_class_kwargs) 


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, 
                output_attentions= None,
                output_hidden_states= None,
                return_dict= None,
                **kwargs):
        # Use model body to get encoder representations
        # the only ones we need for now are input_ids and attention_mask
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        outputs = self.body_model(input_ids, 
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids, 
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict,
                                  **kwargs)
        
        hidden_states = outputs['hidden_states'] # tuples with len 13 (number of layer/block)
        # each with shape: (bs,seq_len,hidden_size_len), e.g. for phobert: (bs,256, 768)
        # Note: hidden_size_len = embedding_size
        
        hidden_concat = torch.cat([hidden_states[i][:,0] for i in range(-1,-self.layer2concat-1,-1)],
                                  -1)
        
        # classification head
        logits = self.classification_head(hidden_concat)

        loss = None
        if labels is not None:
            # labels shape: (bs,2), first is L1, second is L2
            labels_l1 = labels[:,0].view(-1) #(bs,)
            labels_l2 = labels[:,1].view(-1) #(bs,)
            l1_1hot = torch.nn.functional.one_hot(labels_l1, num_classes=self.size_l1)
            l2_1hot = torch.nn.functional.one_hot(labels_l2, num_classes=self.size_l2)
            label_concat_1hot = torch.cat((l1_1hot,l2_1hot),1) # (bs,L1+L2)

            # the original approach: positives and other children of same parents
            _mask = self.standard_mask[labels_l1]
            loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

            loss = loss_func(logits,label_concat_1hot.float())
            loss = torch.mul(loss,_mask)
            loss = (loss.sum(axis=1)/_mask.sum(axis=1)).mean()
            
        # Return model output object
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                     hidden_states=None,
                                     attentions=outputs.attentions)

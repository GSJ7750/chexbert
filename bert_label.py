#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from collections import OrderedDict
from constants import *
from tqdm import tqdm
from types import SimpleNamespace

#unlabeld_dataset
from torch.utils.data import Dataset, DataLoader

#utils
import copy
import json
from sklearn.metrics import f1_score, confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa
from constants import *

#bert labeler
from transformers import BertModel, AutoModel

#bert_tokenizer
from transformers import BertTokenizer, AutoTokenizer
import argparse


# In[43]:


#bert_tokenizer
class bert_tokenizer:
    def get_impressions_from_csv(data):
            #df = pd.read_csv(data)'
            df = pd.DataFrame(data, columns=['Report Impression'])
            imp = df['Report Impression']
            imp = imp.str.strip()
            imp = imp.replace('\n',' ', regex=True)
            imp = imp.replace('\s+', ' ', regex=True)
            imp = imp.str.strip()
            return imp

    def tokenize(impressions, tokenizer):
            new_impressions = []
            print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
            for i in tqdm(range(impressions.shape[0])):
                    tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
                    if tokenized_imp: #not an empty report
                            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
                            if len(res) > 512: #length exceeds maximum size
                                    #print("report length bigger than 512")
                                    res = res[:511] + [tokenizer.sep_token_id]
                            new_impressions.append(res)
                    else: #an empty report
                            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
            return new_impressions

    def load_list(path):
            with open(path, 'r') as filehandle:
                    impressions = json.load(filehandle)
                    return impressions


# In[31]:


#bert labeler
class bert_labeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with 
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(p)
        #size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        #classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        #classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape 
                                            (batch_size, 4) and the last has shape (batch_size, 2)  
        """
        #shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        #shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out


# In[32]:


#utils
class utils:
    def get_weighted_f1_weights(train_path_or_csv):
        """Compute weights used to obtain the weighted average of
           mention, negation and uncertain f1 scores. 
        @param train_path_or_csv: A path to the csv file or a dataframe

        @return weight_dict (dictionary): maps conditions to a list of weights, the order
                                          in the lists is negation, uncertain, positive 
        """
        if isinstance(train_path_or_csv, str):
            df = pd.read_csv(train_path_or_csv)
        else:
            df = train_path_or_csv    
        df.replace(0, 2, inplace=True)
        df.replace(-1, 3, inplace=True)
        df.fillna(0, inplace=True)

        weight_dict = {}
        for cond in CONDITIONS:
            weights = []
            col = df[cond]

            mask = col == 2
            weights.append(mask.sum())

            mask = col == 3
            weights.append(mask.sum())

            mask = col == 1
            weights.append(mask.sum())

            if np.sum(weights) > 0:
                weights = np.array(weights)/np.sum(weights)
            weight_dict[cond] = weights
        return weight_dict

    def weighted_avg(scores, weights):
        """Compute weighted average of scores
        @param scores(List): the task scores
        @param weights (List): corresponding normalized weights

        @return (float): the weighted average of task scores
        """
        return np.sum(np.array(scores) * np.array(weights))

    def compute_train_weights(train_path):
        """Compute class weights for rebalancing rare classes
        @param train_path (str): A path to the training csv file

        @returns weight_arr (torch.Tensor): Tensor of shape (train_set_size), containing
                                            the weight assigned to each training example 
        """
        df = pd.read_csv(train_path)
        cond_weights = {}
        for cond in CONDITIONS:
            col = df[cond]
            val_counts = col.value_counts()
            if cond != 'No Finding':
                weights = {}
                weights['0.0'] = len(df) / val_counts[0]
                weights['-1.0'] = len(df) / val_counts[-1]
                weights['1.0'] = len(df) / val_counts[1]
                weights['nan'] = len(df) / (len(df) - val_counts.sum())
            else:
                weights = {}
                weights['1.0'] = len(df) / val_counts[1]
                weights['nan'] = len(df) / (len(df) - val_counts.sum())

            cond_weights[cond] = weights

        weight_arr = torch.zeros(len(df))
        for i in range(len(df)):     #loop over training set
            for cond in CONDITIONS:  #loop over all conditions
                label = str(df[cond].iloc[i])
                weight_arr[i] += cond_weights[cond][label] #add weight for given class' label

        return weight_arr

    def generate_attention_masks(batch, source_lengths, device):
        """Generate masks for padded batches to avoid self-attention over pad tokens
        @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                               where max_len is length of longest sequence in the batch
        @param source_lengths (List[Int]): List of actual lengths for each of the
                               sequences in the batch
        @param device (torch.device): device on which data should be

        @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
        """
        masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
        for idx, src_len in enumerate(source_lengths):
            masks[idx, src_len:] = 0
        return masks.to(device)

    def compute_mention_f1(y_true, y_pred):
        """Compute the mention F1 score as in CheXpert paper
        @param y_true (list): List of 14 tensors each of shape (dev_set_size)
        @param y_pred (list): Same as y_true but for model predictions

        @returns res (list): List of 14 scalars
        """
        for j in range(len(y_true)):
            y_true[j][y_true[j] == 2] = 1
            y_true[j][y_true[j] == 3] = 1
            y_pred[j][y_pred[j] == 2] = 1
            y_pred[j][y_pred[j] == 3] = 1

        res = []
        for j in range(len(y_true)): 
            res.append(f1_score(y_true[j], y_pred[j], pos_label=1))

        return res

    def compute_blank_f1(y_true, y_pred):
        """Compute the blank F1 score 
        @param y_true (list): List of 14 tensors each of shape (dev_set_size)
        @param y_pred (list): Same as y_true but for model predictions

        @returns res (list): List of 14 scalars                           
        """
        for j in range(len(y_true)):
            y_true[j][y_true[j] == 2] = 1
            y_true[j][y_true[j] == 3] = 1
            y_pred[j][y_pred[j] == 2] = 1
            y_pred[j][y_pred[j] == 3] = 1

        res = []
        for j in range(len(y_true)):
            res.append(f1_score(y_true[j], y_pred[j], pos_label=0))

        return res

    def compute_negation_f1(y_true, y_pred):
        """Compute the negation F1 score as in CheXpert paper
        @param y_true (list): List of 14 tensors each of shape (dev_set_size)
        @param y_pred (list): Same as y_true but for model predictions   

        @returns res (list): List of 14 scalars
        """
        for j in range(len(y_true)):
            y_true[j][y_true[j] == 3] = 0
            y_true[j][y_true[j] == 1] = 0
            y_pred[j][y_pred[j] == 3] = 0
            y_pred[j][y_pred[j] == 1] = 0

        res = []
        for j in range(len(y_true)-1):
            res.append(f1_score(y_true[j], y_pred[j], pos_label=2))

        res.append(0) #No Finding gets score of zero
        return res

    def compute_positive_f1(y_true, y_pred):
        """Compute the positive F1 score
        @param y_true (list): List of 14 tensors each of shape (dev_set_size)
        @param y_pred (list): Same as y_true but for model predictions 

        @returns res (list): List of 14 scalars
        """
        for j in range(len(y_true)):
            y_true[j][y_true[j] == 3] = 0
            y_true[j][y_true[j] == 2] = 0
            y_pred[j][y_pred[j] == 3] = 0
            y_pred[j][y_pred[j] == 2] = 0

        res = []
        for j in range(len(y_true)):
            res.append(f1_score(y_true[j], y_pred[j], pos_label=1))

        return res

    def compute_uncertain_f1(y_true, y_pred):
        """Compute the negation F1 score as in CheXpert paper
        @param y_true (list): List of 14 tensors each of shape (dev_set_size)
        @param y_pred (list): Same as y_true but for model predictions

        @returns res (list): List of 14 scalars
        """
        for j in range(len(y_true)):
            y_true[j][y_true[j] == 2] = 0
            y_true[j][y_true[j] == 1] = 0
            y_pred[j][y_pred[j] == 2] = 0
            y_pred[j][y_pred[j] == 1] = 0

        res = []
        for j in range(len(y_true)-1):
            res.append(f1_score(y_true[j], y_pred[j], pos_label=3))

        res.append(0) #No Finding gets a score of zero
        return res

    def evaluate(model, dev_loader, device, f1_weights, return_pred=False):
        """ Function to evaluate the current model weights
        @param model (nn.Module): the labeler module 
        @param dev_loader (torch.utils.data.DataLoader): dataloader for dev set  
        @param device (torch.device): device on which data should be
        @param f1_weights (dictionary): dictionary mapping conditions to f1
                                        task weights
        @param return_pred (bool): whether to return predictions or not

        @returns res_dict (dictionary): dictionary with keys 'blank', 'mention', 'negation',
                               'uncertain', 'positive' and 'weighted', with values 
                                being lists of length 14 with each element in the 
                                lists as a scalar. If return_pred is true then a 
                                tuple is returned with the aforementioned dictionary 
                                as the first item, a list of predictions as the 
                                second item, and a list of ground truth as the 
                                third item
        """

        was_training = model.training
        model.eval()
        y_pred = [[] for _ in range(len(CONDITIONS))]
        y_true = [[] for _ in range(len(CONDITIONS))]

        with torch.no_grad():
            for i, data in enumerate(dev_loader, 0):
                batch = data['imp'] #(batch_size, max_len)
                batch = batch.to(device)
                label = data['label'] #(batch_size, 14)
                label = label.permute(1, 0).to(device)
                src_len = data['len']
                batch_size = batch.shape[0]
                attn_mask = generate_attention_masks(batch, src_len, device)

                out = model(batch, attn_mask)

                for j in range(len(out)):
                    out[j] = out[j].to('cpu') #move to cpu for sklearn
                    curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                    y_pred[j].append(curr_y_pred)
                    y_true[j].append(label[j].to('cpu'))

                if (i+1) % 200 == 0:
                    print('Evaluation batch no: ', i+1)

        for j in range(len(y_true)):
            y_true[j] = torch.cat(y_true[j], dim=0)
            y_pred[j] = torch.cat(y_pred[j], dim=0)

        if was_training:
            model.train()

        mention_f1 = compute_mention_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
        negation_f1 = compute_negation_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
        uncertain_f1 = compute_uncertain_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
        positive_f1 = compute_positive_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
        blank_f1 = compute_blank_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))

        weighted = []
        kappas = []
        for j in range(len(y_pred)):
            cond = CONDITIONS[j]
            avg = weighted_avg([negation_f1[j], uncertain_f1[j], positive_f1[j]], f1_weights[cond])
            weighted.append(avg)

            mat = confusion_matrix(y_true[j], y_pred[j])
            kappas.append(cohens_kappa(mat, return_results=False))

        res_dict = {'mention': mention_f1,
                    'blank': blank_f1,
                    'negation': negation_f1,
                    'uncertain': uncertain_f1,
                    'positive': positive_f1,
                    'weighted': weighted,
                    'kappa': kappas}

        if return_pred:
            return res_dict, y_pred, y_true
        else:
            return res_dict

    def test(model, checkpoint_path, test_ld, f1_weights):
        """Evaluate model on test set. 
        @param model (nn.Module): labeler module
        @param checkpoint_path (string): location of saved model checkpoint
        @param test_ld (dataloader): dataloader for test set
        @param f1_weights (dictionary): maps conditions to f1 task weights
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print("Doing evaluation on test set\n")
        metrics = evaluate(model, test_ld, device, f1_weights)
        weighted = metrics['weighted']
        kappas = metrics['kappa']

        for j in range(len(CONDITIONS)):
            print('%s kappa: %.3f' % (CONDITIONS[j], kappas[j]))
        print('average: %.3f' % np.mean(kappas))

        print()
        for j in range(len(CONDITIONS)):
            print('%s weighted_f1: %.3f' % (CONDITIONS[j], weighted[j]))
        print('average of weighted_f1: %.3f' % (np.mean(weighted)))

        print()
        for j in range(len(CONDITIONS)):
            print('%s blank_f1:  %.3f, negation_f1: %.3f, uncertain_f1: %.3f, positive_f1: %.3f' % (CONDITIONS[j],
                                                                                                    metrics['blank'][j],
                                                                                                    metrics['negation'][j],
                                                                                                    metrics['uncertain'][j],
                                                                                                    metrics['positive'][j]))

        men_macro_avg = np.mean(metrics['mention'])
        neg_macro_avg = np.mean(metrics['negation'][:-1]) #No Finding has no negations
        unc_macro_avg = np.mean(metrics['uncertain'][:-2]) #No Finding, Support Devices have no uncertain labels in test set
        pos_macro_avg = np.mean(metrics['positive'])
        blank_macro_avg = np.mean(metrics['blank'])

        print("blank macro avg: %.3f, negation macro avg: %.3f, uncertain macro avg: %.3f, positive macro avg: %.3f" % (blank_macro_avg,
                                                                                                                        neg_macro_avg,
                                                                                                                        unc_macro_avg,
                                                                                                                        pos_macro_avg))
        print()
        for j in range(len(CONDITIONS)):
            print('%s mention_f1: %.3f' % (CONDITIONS[j], metrics['mention'][j]))
        print('mention macro avg: %.3f' % men_macro_avg)


    def label_report_list(checkpoint_path, report_list):
        """ Evaluate model on list of reports.
        @param checkpoint_path (string): location of saved model checkpoint
        @param report_list (list): list of report impressions (string)
        """
        imp = pd.Series(report_list)
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('[0-9]\.', '', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()

        model = bert_labeler()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        y_pred = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        new_imps = tokenize(imp, tokenizer)
        with torch.no_grad():
            for imp in new_imps:
                # run forward prop
                imp = torch.LongTensor(imp)
                source = imp.view(1, len(imp))

                attention = torch.ones(len(imp))
                attention = attention.view(1, len(imp))
                out = model(source.to(device), attention.to(device))

                # get predictions
                result = {}
                for j in range(len(out)):
                    curr_y_pred = out[j].argmax(dim=1) #shape is (1)
                    result[CONDITIONS[j]] = CLASS_MAPPING[curr_y_pred.item()]
                y_pred.append(result)
        return y_pred


# In[33]:


#unlabeld_dataset


class UnlabeledDataset(Dataset):
        """The dataset to contain report impressions without any labels."""
        
        def __init__(self, csv_path):
                """ Initialize the dataset object
                @param csv_path (string): path to the csv file containing rhe reports. It
                                          should have a column named "Report Impression"
                """
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                impressions = bert_tokenizer.get_impressions_from_csv(csv_path)
                self.encoded_imp = bert_tokenizer.tokenize(impressions, tokenizer)

        def __len__(self):
                """Compute the length of the dataset

                @return (int): size of the dataframe
                """
                return len(self.encoded_imp)

        def __getitem__(self, idx):
                """ Functionality to index into the dataset
                @param idx (int): Integer index into the dataset

                @return (dictionary): Has keys 'imp', 'label' and 'len'. The value of 'imp' is
                                      a LongTensor of an encoded impression. The value of 'label'
                                      is a LongTensor containing the labels and 'the value of
                                      'len' is an integer representing the length of imp's value
                """
                if torch.is_tensor(idx):
                        idx = idx.tolist()
                imp = self.encoded_imp[idx]
                imp = torch.LongTensor(imp)
                return {"imp": imp, "len": imp.shape[0]}


# In[51]:


#------- Main(label.py)
BATCH_SIZE = 32
NUM_WORKERS = 0
PAD_IDX = 0
CONDITIONS = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices','No Finding']

def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch

def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader
    
def label(checkpoint_path, csv_path):
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld = load_unlabeled_data(csv_path)
    
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)
            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                print(len(out), j, curr_y_pred)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)
             
    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    return y_pred

def save_preds(y_pred, csv_path, out_path):
    """Save predictions as out_path/labeled_reports.csv 
    @param y_pred (List[List[int]]): list of predictions for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    
    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    reports = pd.read_csv(csv_path)['Report Impression']

    df['Report Impression'] = reports.tolist()
    new_cols = ['Report Impression'] + CONDITIONS
    df = df[new_cols]

    df.replace(0, np.nan, inplace=True) #blank class is NaN
    df.replace(3, -1, inplace=True)     #uncertain class is -1
    df.replace(2, 0, inplace=True)      #negative class is 0 
    
    df.to_csv(os.path.join(out_path, 'labeled_reports.csv'), index=False)
    

    
def set_flags(**args):
            flag = SimpleNamespace(**args)
            return flag
        

def run(report, ckpt_path):
    data = [report]
    ckpt = ckpt_path
    
    y_pred = label(ckpt, data)
    
    report = pd.DataFrame(data, columns=['Report Impression'])
    result = np.asarray(y_pred).T
    df = pd.DataFrame(result, columns=CONDITIONS)
    df = report.join(df)
    
    
    return df


# In[52]:


#run('sample_reports', 'ckpt/chexbert.pth')


# In[ ]:





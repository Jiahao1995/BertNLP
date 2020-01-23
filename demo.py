import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

def init_bert(pre_trained='bert-base-cased', num_labels=2):
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(pre_trained)

    # 获得hidden states
    model = BertForSequenceClassification.from_pretrained(pre_trained, num_labels=num_labels)
    model.eval()

    # print('成功加载bert模型')

    return tokenizer, model

def make_bert(tokenizer, model, text):
    # "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]" --> ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)

    # 将token转换为index
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # [CLS]到[SEP]是0，[SEP]+1到[SEP]是1
    segments_ids = []
    flag = 0
    for token in indexed_tokens:
        if flag == 0:
            segments_ids.append(0)
        if flag == 1:
            segments_ids.append(1)
        if token == 102: # [SEP]
            flag = 1

    # 将输入转换为tensor
    input_ids = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    input_mask = torch.ones_like(input_ids)

    # 预测每一层的hidden states
    # print(input_ids, segments_tensors, input_mask)
    with torch.no_grad():
        logit = model(input_ids, segments_tensors, input_mask)

    return logit

text1 = 'Who was Jim Henson ?'
text2 = 'Jim Henson was a puppeteer'
text = '[CLS] ' + text1 + ' [SEP] ' + text2 + ' [SEP]'

tokenizer, model = init_bert(pre_trained='outputs/MRPC', num_labels=2)
logit = make_bert(tokenizer, model, text)
print(logit[0])

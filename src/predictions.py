from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
from sklearn.metrics import classification_report


roberta = RobertaModel.from_pretrained(
    '/data/szr207/projects/SeerBERT/RoBERTa/fairseq/checkpoints',
    checkpoint_file='checkpoint7.pt',
    data_name_or_path='QQP-bin'
)
ncorrect, nsamples, pred, actual = [], [], [], []
roberta.eval()
count = 11000
with open('/data/szr207/projects/SeerBERT/RoBERTa/fairseq/glue_data/QQP/test.tsv') as fin:
    for index, line in tqdm(enumerate(fin)):
        try:
            tokens = line.strip().split('\t')
            qid, aid, sent1, sent2, target = tokens[1], tokens[2], tokens[3], tokens[4], tokens[5]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
#             print(prediction)
#             if prediction==1:
#                 print(qid, aid)
            actual.append(int(target))
            pred.append(prediction)
            count-=1
            if count<0:
                break
#             prediction_label = label_fn(prediction)
        except:
            continue
            
            
print(classification_report(actual, pred))
        
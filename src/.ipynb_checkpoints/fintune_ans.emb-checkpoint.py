import glob
from tqdm import tqdm
import os
import jsonlines
import re       
from transformers import pipeline
import torch
import numpy as np


root_path = '/data/szr207/dataset/ArqMath/jsons/answers/'

list_ans = []
full_meta = [] 
with jsonlines.open(os.path.join(root_path,'all.ans.jsonl')) as reader:
        for obj in tqdm(reader):
            if obj['body']:
                obj['body'] = re.sub('<[^<]+?>', '',  obj['body'])
                list_ans.append(obj['body'])
            full_meta.append(obj)

pro_ans = []

for i in tqdm(list_ans):
    if i.rstrip() == '':
        pro_ans.append('NONE')
    else:
        pro_ans.append(i)

feat_ext = pipeline("feature-extraction", model="shauryr/arqmath-roberta-base", tokenizer='roberta-base', device=0)
ans_emb = []
for ans in tqdm(pro_ans):
    feat = feat_ext(ans)[0][0]
    ans_emb.append(feat)

np.save('fintune.ans.emb.npy', ans_emb)
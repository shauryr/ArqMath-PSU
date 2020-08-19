# PSU's entry for ARQMath CLEF 2020 Task

notebooks -
- analysis : analysis of if answers or questions were enough to get the relevant documents. Turns out answer index and question index give non overlapping relevant answers. So both are important
- anerini_format : this notebook converts the ArqMath data to Anserini format
- anserini+ARQBert : code to highlight relevant parts of the retrieved posts
- BERT_answers : code to retreive similar answers using BERT embeddings - turns out it is really bad
- bpe_math : sentencepiece and tf-idf applied to LateX of math formula
- buffer : you know to run random code
- create_pairwise : code to generate data in QQP format for training for relevance
- data_stats : code for ARQMath EDA
- final_sub-rrf : final code for submission in the ARQMATH CLEF 2020 task with RRF
- final_sub : Actual final code for submission
- LDA : nothing very interesting here - visualization for topics in the data
- post_reader : initial starter code
- query_es : ES baseline
- ques_ans_index-reranker : Searching more like this posts in the ques+ans index - yeilding the best results on the sample queries
- ques_ans_index : code to index ques+ans in bulk
- question_cluster : using bert-large and faiss to check question clusters
- ranking_questions : why ranking questions makes sense
- Reranker : first version of code where reranking was discovered to help boost up ndcg
- result_highlight : just to see what is relevant in the answers according to BERT
- sim_questions-reranker : similar to sim_questions - this is where searching both ques and answers as one document in index was discovered.
- testing_finetune : Testing the performance of ARQBert


Findings :
- ElasticSearch will give preference to text when retreiving documents - instead of math formulae - we need a high recall formula search technique - BPE ? will it work ? how to include it ? Right now the system will perform really well with text dominant queries

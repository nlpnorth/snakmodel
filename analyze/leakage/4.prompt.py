from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

token = open('hf_token').readlines()[0].strip()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#lm = 'meta-llama/Llama-2-7b-hf'
lm = 'meta-llama/Llama-2-7b-chat-hf'
lang_model = AutoModelForCausalLM.from_pretrained(lm, token=token)
lang_model.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False, token=token)


datasets = ['AngryTweets', 'DANSK', 'Danish Citizenship tests', 'Danish HellaSwag', 'Nordjylland news summarization', 'ScaLA']
datafiles = ['DDSC/angry-tweets-test.conll', 'chcaa/dansk-ner-test.conll', 'alexandrainst/danish-citizen-tests-train.conll', 'alexandrainst/m_hellaswag-da-val.conll', 'alexandrainst/nordjylland-news-summarization-test.conll', 'alexandrainst/scala-da-test.conll']#, 'alexandrainst/scandi-qa-da-test.conll']:
for dataset, datafile in zip(datasets, datafiles):
    print('====================================')
    print(dataset)
    prompts = ["What is the task that is tackled in the " + dataset + " benchmark?", "What is the task that is tackled in the " + dataset + " dataset?", "What are the first 2 sentences in the " + dataset + " dataset?"]
    for prompt in prompts:
        tokked = tokenizer(prompt, return_tensors='pt')['input_ids']
        tokked = tokked.to(DEVICE)
        generated_ids = lang_model.generate(tokked, max_new_tokens=30)
        tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print('prompt: ' + prompt)
        print('output: ' + ' '.join(tokens).replace('\n', ' ').replace(prompt, ''))

    print()
    idx = 0
    prompt = ''
    while len(prompt) < 20:
        prompt = open(datafile).readlines()[idx]
        idx += 1
    gold_answer = open(datafile).readlines()[idx]
    tokked = tokenizer(prompt, return_tensors='pt')['input_ids']
    tokked = tokked.to(DEVICE)
    generated_ids = lang_model.generate(tokked, max_new_tokens=40)
    tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print('Next sent: ' + gold_answer[:100].replace('\n', ' '))
    print('Pred. next sent: ' + ' '.join(tokens)[:100].replace('\n', ' '))

    middle = int(len(tokked[0])/2)
    tokked_start = tokked[:,:middle]
    tokked_end = [tokked[:,middle:]]
    generated_ids = lang_model.generate(tokked_start, max_new_tokens=30)
    pred_end = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    gold_end = tokenizer.batch_decode(tokked_end[0], skip_special_tokens=True)
    print('Next words: ' + gold_end[0][:30].replace('\n', ' '))
    print('Pred next words: ' + pred_end[0][:30].replace('\n', ' ' ))
    print('\n')
                                                                                                                                                                                                                                                                                         


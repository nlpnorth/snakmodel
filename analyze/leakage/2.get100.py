import random
seed = 8446
random.seed(seed)
import os
import nltk


if not os.path.isdir('ngrams.' + str(seed)):
    os.mkdir('ngrams.' + str(seed))
cmd = ''
for datafile in ['DDSC/angry-tweets-test.conll', 'chcaa/dansk-ner-test.conll', 'alexandrainst/danish-citizen-tests-train.conll', 'alexandrainst/m_hellaswag-da-val.conll', 'alexandrainst/nordjylland-news-summarization-test.conll', 'alexandrainst/scala-da-test.conll']:#, 'alexandrainst/scandi-qa-da-test.conll']:
    print(datafile)
    data = open(datafile).readlines()
    tok = data[0].strip().split('\t')
    text_idx = 0
    if tok[1].count(' ') > tok[0].count(' '):
        text_idx = 1
    if 'ner' in datafile:
        text_idx = 0
    random.shuffle(data)
    outFile = open('ngrams.' + str(seed) + '/' + datafile.split('/')[-1] + '.100', 'w')
    for line in data[:200]:
        tok = line.strip().split('\t')
        sents = nltk.sent_tokenize(tok[text_idx])
        if len(sents) == 0:
            continue
        random.shuffle(sents)
        
        cleaned = sents[0].replace('@USER', '').replace('[LINK]', '').strip().replace('  ', ' ').replace('"', '\\"').replace('[header]', '').replace('[step]', '').replace('[title]', '').replace('|', '').replace('!', '\!').replace('(', '\(').replace(')', '\)').strip()
        cleaned = ' '.join(cleaned.split(' ')[:8])
        if len(cleaned.split(' ')) < 6:
            continue
        if cleaned[-1] == '.':
            cleaned = cleaned[:-1]
        outFile.write(cleaned + '\n')
        cmd += cleaned + '|' 
    outFile.close()

print('grep -E "' + cmd[:-1]+ '" ../_raw_data/gigaword/*danish > results.' + str(seed))


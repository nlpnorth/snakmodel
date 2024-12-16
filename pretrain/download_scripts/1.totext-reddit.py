import json

outFile = open('data/reddit.da/utterances.txt', 'w')
for line in open('data/reddit.da/utterances.jsonl'):
    txt = json.loads(line)['text'].strip()
    if len(txt) < 3 or txt in ['[deleted]', 'removed']:
        continue
    
    outFile.write(txt + '\n')

outFile.close()

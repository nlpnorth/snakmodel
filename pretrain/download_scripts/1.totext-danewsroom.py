import json

outFile = open('data/danewsroom/danewsroom.txt', 'w')
for line in open('data/danewsroom/danewsroom.jsonl'):
    txt = json.loads(line)['text'].strip()
    if len(txt) < 3 or txt in ['[deleted]', 'removed']:
        continue
    
    outFile.write(txt + '\n')
outFile.close()



from datasets import load_dataset

ds = load_dataset("uonlp/CulturaX",  'da', use_auth_token=True, streaming=True)

trainsplit = ds['train']
outFile = open('culturaX.txt', 'w')
for item in trainsplit:
    outFile.write(item['text'] + '\n')
outFile.close()


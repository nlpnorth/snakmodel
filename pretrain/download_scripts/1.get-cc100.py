from datasets import load_dataset

ds = load_dataset("statmt/cc100",  'da', streaming=True, trust_remote_code=True)

trainsplit = ds['train']
outFile = open('cc-100-da.txt', 'w')
for item in trainsplit:
    outFile.write(item['text'] + '\n')
outFile.close()


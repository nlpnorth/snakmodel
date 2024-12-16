import os

rootdir = 'data/gigaword/dagw/sektioner/'


outfile = open('data/gigaword/all.txt', 'w')
for (root,dirs,files) in os.walk(rootdir, topdown=True): 
    for txtfile in files:
        if txtfile.endswith('jsonl') or txtfile.endswith('json'):
            continue
        if 'readme' in txtfile.lower() or txtfile == 'LICENSE':
            continue
        print(txtfile)
        for line in open(root + '/' + txtfile):
            if len(line) > 2:
                outfile.write(line)
outfile.close()

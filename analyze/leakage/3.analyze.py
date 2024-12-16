import os
seed = '513'
outfile = open('results.' + seed, 'w')
for dataset in os.listdir('ngrams.' + seed):
    print(dataset)
    outfile.write(dataset + '\n')
    for line in open('ngrams.' + seed + '/' + dataset):
        query = line.strip()
        found = 0
        for line2 in open('out.' + seed):
            if query in line2:
                found +=1
        if found > 0:
            outfile.write(str(found) + '\t' + query + '\n')
    print()
    outfile.write('\n')
outfile.close()


import myutils
import fasttext
import os

sf = myutils.ScriptFinder()

if not os.path.isfile('lid.176.bin'):
    os.system('wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin')

model = fasttext.load_model("lid.176.bin")
# 'ÆØÅß' = all Latin
#for char in 'ÆØÅß':
#    print(sf.find_char(char))

for txtFile in myutils.txtFiles:
    outpath = txtFile + '.danish'
    filteredpath = txtFile + '.other'
    if not os.path.isfile(outpath):
    #if True:
        print('Processing ' + txtFile + ': ', end='', flush=True)
        outfile = open(outpath, 'w')
        filteredfile = open(filteredpath, 'w')
        newsize = 0
        for lineIdx, line in enumerate(open(txtFile)):
            line = line.strip()
            #total_latin = sum([sf.find_char(char) == 'Latin' for char in line])
            #if len(line) > 0 and total_latin/len(line) >= .5:
        
            #total_latin = sum([sf.find_char(char) == 'Latin' for char in line[:10]])
            if True:
                label, prob = model.predict(line)
                if label[0] == '__label__da' and prob[0] > .6:
                    outfile.write(line + '\n')
                    newsize+=1
                else:
                    filteredfile.write(line + '\n')
            if lineIdx % 500000 == 0:
                print('.', end='', flush=True)
        print()
        print('Kept:')
        print(newsize)
        print('Original:')
        print(lineIdx)
        print()
        outfile.close()
        filteredfile.close()


# Check for script or not (full data: 1,838,617) 
#                 not         first 10 chars     full text
# time            1 minute    1 minute           3 minutes
# #instances      1,419,348   1,409,690          1,418,978



import os
import sys
import bz2
import json
import fasttext

if not os.path.isfile('lid.176.bin'):
    os.system('wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin') 
model = fasttext.load_model('lid.176.bin')

def processFile(minuteFile, outFile):
    try:
        with bz2.open(minuteFile, "rt") as bz_file:
            for line in bz_file:
                try:
                    tweetData= json.loads(line)
                    user = tweetData['user']['screen_name']
                    text = tweetData['text'].replace('\n', ' ')
                    if 'lang' in tweetData:
                        tw_lang = tweetData['lang']
                    else:
                        tw_lang = tweetData['user']['lang']
                    prediction = model.predict(text)
                    fast_lang = prediction[0][0][-2:]
                    conf = str(round(prediction[1][0], 4))
                    if tw_lang == None:
                        tw_lang = ''

                    outFile.write('\t'.join([tw_lang, fast_lang, conf, user, text]) + '\n')
                except:
                    if 'delete' not in line:
                        print('delete not in line: ', line)
                    pass
    except:
        print('incorrect format: ' + minuteFile)
        


def mkDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

yearDir = 'full/' + sys.argv[1]

mkDir('txt')
yearDirOut = 'txt/' + sys.argv[1]
mkDir(yearDirOut)
#2017/01/20/02

for monthDir in os.listdir(yearDir):
    monthDirOut = yearDirOut + '/' + monthDir
    monthDir = yearDir + '/' + monthDir
    mkDir(monthDirOut)
    for dayDir in os.listdir(monthDir):
        dayDirOut = monthDirOut + '/' + dayDir
        dayDir = monthDir + '/' + dayDir
        mkDir(dayDirOut)
        for hourDir in os.listdir(dayDir):
            hourPath = dayDirOut + '/' + hourDir + '.txt'
            hourDir = dayDir + '/' + hourDir
            if os.path.isfile(hourPath + '.gz') or os.path.isfile(hourDir):
                continue
            hourFile = open(hourPath, 'w')
            for minuteFile in os.listdir(hourDir):
                minuteFile = hourDir + '/' + minuteFile
                processFile(minuteFile, hourFile)
            hourFile.close()
            os.system('gzip ' + hourPath)




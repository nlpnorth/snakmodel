mkdir -p data

# bookshop
cd data
mkdir -p bookshop
cd bookshop
wget https://object.pouta.csc.fi/OPUS-EUbookshop/v2/mono/da.txt.gz
gunzip da.txt.gz
cd ../

# cc100
mkdir -p cc100
cd cc100
python3 ../../data_scripts/get-c100.py
cd ../

# culturax
mkdir -p culturax
cd culturax
python3 ../../data_scripts/get-culturax.py
cd ../

# danewsroom
mkdir -p danewsroom
echo "first obtain danewsroom.jsonl from the original publication, and put in data/danewsroom/danewsroom.jsonl"
cd ../
python3 data_scripts/1.totext-danewsroom.py
cd data

# dawiki
echo "note that this gets the latest wikipedia dump, we used the one from jan. 2024"
mkdir -p dawiki
cd ../
./data_scripts/wiki.sh da

# ftspeech
mkdir -p ftspeech
echo "first obtain ftspeech.tar.gz from the original publication, and put in data/ftspeech/ftspeech.tar.gz"
cd ftspeech
tar -xvf ftspeech.tar.gz ftspeech/lm/ft_lm_train_data.txt
cd ../

# gigaword
mkdir -p gigaword
echo "first obtain danishgigaword10.zip from the original publication, and put in data/gigaword/danishgigaword10.zip"
cd ../
python3 data_scripts/1.totext-gigaword.py
cd data

# Opensubtitles
mkdir -p opensubtitles
cd opensubtitles
wget https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/da.txt.gz
gunzip da.txt.gz
cd ../

# reddit.da
mkdir -p reddit.da
cd reddit.da
wget https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/DeepFriedMemesCSS~-~Denmark/Denmark.corpus.zip
unzip Denmark.corpus.zip
python3 data_scripts/1.totext-reddit.py

# twitter
echo "for instructions on how to get the Twitter data see data_scripts/twitter.txt, note that it took me a couple of months"

# language filtering
cd ../
python3 scripts/2.langid.py


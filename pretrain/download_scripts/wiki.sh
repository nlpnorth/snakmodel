#!/bin/sh
set -e

LG=$1
WIKI_DUMP_NAME=${LG}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_DOWNLOAD_URL=https://dumps.wikimedia.org/${LG}wiki/latest/$WIKI_DUMP_NAME

# download latest Wikipedia dump in chosen language
echo "Downloading the latest $LG-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
wget -c $WIKI_DUMP_DOWNLOAD_URL
echo "Succesfully downloaded the latest $LG-language Wikipedia dump to $WIKI_DUMP_NAME"

WIKI_DUMP_FILE_IN=$WIKI_DUMP_NAME
WIKI_DUMP_FILE_OUT=${WIKI_DUMP_FILE_IN%%.*}.txt

# install WikiExtractor repository
if [ ! -d wikiextractor ]; then
    pip3 install git+https://github.com/attardi/wikiextractor.git@ab8988ebfa9e4557411f3d4c0f4ccda139e18875 -t wikiextractor
fi

# extract and clean the chosen Wikipedia dump
echo "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."
cd wikiextractor
python3 -m wikiextractor.WikiExtractor ../$WIKI_DUMP_FILE_IN --processes 8 -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
echo "Succesfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"
cd ../

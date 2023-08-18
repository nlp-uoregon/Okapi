lang=$1
if [ -z "$lang" ]
then
    wget -r -np -nH --cut-dirs=2 --reject="index.html*" http://nlp.uoregon.edu/download/okapi-data/datasets
else
    wget -P ./datasets/multilingual-alpaca-52k http://nlp.uoregon.edu/download/okapi-data/datasets/multilingual-alpaca-52k/$lang.json
    wget -P ./datasets/multilingual-ranking-data-42k http://nlp.uoregon.edu/download/okapi-data/datasets/multilingual-ranking-data-42k/$lang.json
    wget -P ./datasets/multilingual-rl-tuning-64k http://nlp.uoregon.edu/download/okapi-data/datasets/multilingual-rl-tuning-64k/$lang.json
fi

# Data collection

This folder the scripts that can re-create the full data we used for language model training. All scripts assume to be ran from the `pretrain`` folder, and will collect the data in a folder named `data`. To see all the commands, you can inspect `data_scripts/0.download.sh`. Note that some of the parts of the datasets could not be automated, as the data needs to be obtained from the authors of the papers. Also, this script takes very long (especially the Twitter download can take months), and uses a lot of disk space.


Below, we list the citations for each of the datasets (WHICH YOU SHOULD USE IF YOU USE THE DATA):

Bookshop:
```
@inproceedings{tiedemann-2012-parallel,
    title = "Parallel Data, Tools and Interfaces in {OPUS}",
    author = {Tiedemann, J{\"o}rg},
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Declerck, Thierry  and
      Do{\u{g}}an, Mehmet U{\u{g}}ur  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf",
    pages = "2214--2218",
}
```

CC100
```
@inproceedings{wenzek-etal-2020-ccnet,
    title = "{CCN}et: Extracting High Quality Monolingual Datasets from Web Crawl Data",
    author = "Wenzek, Guillaume  and
      Lachaux, Marie-Anne  and
      Conneau, Alexis  and
      Chaudhary, Vishrav  and
      Guzm{\'a}n, Francisco  and
      Joulin, Armand  and
      Grave, Edouard",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.494",
    pages = "4003--4012",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
```

culturax:
```
@inproceedings{nguyen-etal-2024-culturax,
    title = "{C}ultura{X}: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages",
    author = "Nguyen, Thuat  and
      Nguyen, Chien Van  and
      Lai, Viet Dac  and
      Man, Hieu  and
      Ngo, Nghia Trung  and
      Dernoncourt, Franck  and
      Rossi, Ryan A.  and
      Nguyen, Thien Huu",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.377",
    pages = "4226--4237",
}
```

danewsroom:
```
@inproceedings{varab-schluter-2020-danewsroom,
    title = "{D}a{N}ewsroom: A Large-scale {D}anish Summarisation Dataset",
    author = "Varab, Daniel  and
      Schluter, Natalie",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.831",
    pages = "6731--6739",
    language = "English",
}
```

dawiki:
```
@misc{Wikiextractor2015,
  author = {Giusepppe Attardi},
  title = {WikiExtractor},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/attardi/wikiextractor}}
}
```

ftspeech:
```
@article{kirkedal2020ft,
  title={FT speech: Danish parliament speech corpus},
  author={Kirkedal, Andreas and Stepanovi{\'c}, Marija and Plank, Barbara},
  journal={arXiv preprint arXiv:2005.12368},
  year={2020}
}
```

gigaword:
```
@inproceedings{stromberg-derczynski-etal-2021-danish,
    title = "The {D}anish {G}igaword Corpus",
    author = "Str{\o}mberg-Derczynski, Leon  and
      Ciosici, Manuel  and
      Baglini, Rebekah  and
      Christiansen, Morten H.  and
      Dalsgaard, Jacob Aarup  and
      Fusaroli, Riccardo  and
      Henrichsen, Peter Juel  and
      Hvingelby, Rasmus  and
      Kirkedal, Andreas  and
      Kjeldsen, Alex Speed  and
      Ladefoged, Claus  and
      Nielsen, Finn {\AA}rup  and
      Madsen, Jens  and
      Petersen, Malte Lau  and
      Rystr{\o}m, Jonathan Hvithamar  and
      Varab, Daniel",
    editor = "Dobnik, Simon  and
      {\O}vrelid, Lilja",
    booktitle = "Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may # " 31--2 " # jun,
    year = "2021",
    address = "Reykjavik, Iceland (Online)",
    publisher = {Link{\"o}ping University Electronic Press, Sweden},
    url = "https://aclanthology.org/2021.nodalida-main.46",
    pages = "413--421"
}
```

Opensubstitles:
```
add a link to http://www.opensubtitles.org/ to your website and to your reports and publications produced with the data! 
@inproceedings{lison-tiedemann-2016-opensubtitles2016,
    title = "{O}pen{S}ubtitles2016: Extracting Large Parallel Corpora from Movie and {TV} Subtitles",
    author = {Lison, Pierre  and
      Tiedemann, J{\"o}rg},
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Grobelnik, Marko  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, Helene  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Tenth International Conference on Language Resources and Evaluation ({LREC}'16)",
    month = may,
    year = "2016",
    address = "Portoro{\v{z}}, Slovenia",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L16-1147",
    pages = "923--929",
}
```

ConvoKit:
```
@inproceedings{chang-etal-2020-convokit,
    title = "{C}onvo{K}it: A Toolkit for the Analysis of Conversations",
    author = "Chang, Jonathan P.  and
      Chiam, Caleb  and
      Fu, Liye  and
      Wang, Andrew  and
      Zhang, Justine  and
      Danescu-Niculescu-Mizil, Cristian",
    editor = "Pietquin, Olivier  and
      Muresan, Smaranda  and
      Chen, Vivian  and
      Kennington, Casey  and
      Vandyke, David  and
      Dethlefs, Nina  and
      Inoue, Koji  and
      Ekstedt, Erik  and
      Ultes, Stefan",
    booktitle = "Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.sigdial-1.8",
    doi = "10.18653/v1/2020.sigdial-1.8",
    pages = "57--60",
}
```

Language filtering:
```
@inproceedings{joulin-etal-2017-bag,
    title = "Bag of Tricks for Efficient Text Classification",
    author = "Joulin, Armand  and
      Grave, Edouard  and
      Bojanowski, Piotr  and
      Mikolov, Tomas",
    editor = "Lapata, Mirella  and
      Blunsom, Phil  and
      Koller, Alexander",
    booktitle = "Proceedings of the 15th Conference of the {E}uropean Chapter of the Association for Computational Linguistics: Volume 2, Short Papers",
    month = apr,
    year = "2017",
    address = "Valencia, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/E17-2068",
    pages = "427--431",
    abstract = "This paper explores a simple and efficient baseline for text classification. Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation. We can train fastText on more than one billion words in less than ten minutes using a standard multicore CPU, and classify half a million sentences among 312K classes in less than a minute.",
}
```

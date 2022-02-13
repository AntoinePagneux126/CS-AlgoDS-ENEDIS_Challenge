# Algo data science, challenge ENEDIS & ENS : École CentraleSupelec 2021-2022 

![CentraleSupelec Logo](https://www.centralesupelec.fr/sites/all/themes/cs_theme/medias/common/images/intro/logo_nouveau.jpg)


## Authors 
* Matthieu Briet : matthieu.briet@student-cs.fr
* Tanguy Colleville : tanguy.colleville@student-cs.fr
* Antoine Pagneux : antoine.pagneux@student-cs.fr 

## Useful links 
* Our Workspace : [Here](https://tanguycolleville.notion.site/Algorithms-in-Data-Science-8c88a1d9998e466c9f6f3e35ab03e8c1)
* Our Documentation : [Here](https://www.overleaf.com/project/61feb5dc3d27be675ebfa804)
* Our Video : [Here](https://www.youtube.com/watch?v=L7_RWz8G-yE)


## Summary
  - [Authors ](#authors-)
  - [Useful links](#Useful-links)
  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Our approach](#our--approach)
  - [Architecture & overview](#architecture--overview)
  - [Run Model](#run-model)
  - [Documentation](#documentation)
  - [Conclusion](#conclusion)

 ## Introduction 
  This project is a sligthly modified version of previous ENS Challenge proposed in 2019 by ENEDIS, for more details please click [here](https://challengedata.ens.fr/challenges/6).  

 ## Our approach
   * Data investigation, Point out issue that we may encounter
   * Preprocessing & Features engineering
   * Modeling
   * Monitoring
   * Evaluating
   * Improving

 ## Architecture & overview
 
 ```
.
├── README.md
├── config
│   └── config.ini
├── data
│   ├── inout.csv
│   ├── inputs.csv
│   └── outputs.csv
├── docs
│   ├── Makefile
│   ├── analysis_dataset.html
│   ├── conf.py
│   ├── index.rst
│   └── make.bat
│   └── Report_Challenge_ENEDIS_Briet_Colleville_Pagneux.pdf
├── jobAM.batch
├── notebook
│   ├── Statistical_Models.ipynb
│   ├── data.ipynb
│   ├── notebook_arimax.ipynb
│   ├── notebook_inout_dataviz.ipynb
│   └── prophet_Tanguy.ipynb
├── requirements.txt
├── src
│   ├── arimax.py
│   ├── classifier.py
│   ├── configuration.py
│   ├── mailsender.py
│   ├── main.py
│   ├── models.py
│   ├── prophet.py
│   ├── sarimax.py
│   └── utils.py
└── test
    └── test_merger.py
 ```
Inout.csv is nothing else than the inputs and outputs csv files merged.
 ## Run Model 

* Example to run an ARIMAX to predict the (default) RES1_BASE output: ```python3 main.py --model=ARIMAX```
* Example to run an ARIMAX to predict the RES11_BASE output: ```python3 main.py --model=ARIMAX --timeserie=RES11_BASE```
* Example to run an ARIMAX to predict all outputs: ```python3 main.py --mode=train --model=ARIMAX --timeserie=all```
* Example to run an ARIMAX to predict the (default) RES1_BASE output with p,q,d coefficients: ```python3 main.py --model=ARIMAX --order=1,1,1```

Please note that by providing your email adress in ```config/config.ini``` results will be sended to you by mail as soon as the code ended. 

 ## Documentation 
 * You can access to python code documentation in ```docs/_build/hmtl/index.html```

 * You can access to python coverage report in ```docs/htmlcov/index.html```

 * You can access to python pylint report in ```docs/pylint_report.txt```

 * You can access the 5 pages report about the projet pylint report in ```docs/Report_Challenge_ENEDIS_Briet_Colleville_Pagneux.pdf```


 ## Conclusion 
 Since the problem was complicated, we have learned so much about modeling time series and so on. The lecture was just an appetizer to go further on interesting point go get a better score with our model for this project.



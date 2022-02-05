# Algo data science, challenge ENEDIS & ENS : École CentraleSupelec 2021-2022 

![CentraleSupelec Logo](https://www.centralesupelec.fr/sites/all/themes/cs_theme/medias/common/images/intro/logo_nouveau.jpg)


## Authors 
* Matthieu Briet : matthieu.briet@student-cs.fr
* Tanguy Colleville : tanguy.colleville@student-cs.fr
* Antoine Pagneux : antoine.pagneux@student-cs.fr 

## Useful links 
* Our Workspace : [Here](https://tanguycolleville.notion.site/Algorithms-in-Data-Science-8c88a1d9998e466c9f6f3e35ab03e8c1)
* Our Documentation : [Here](https://www.overleaf.com/project/61feb5dc3d27be675ebfa804)
* Our Video : [Here]()


## Summary
  - [Authors ](#authors-)
  - [Useful links](#Useful-links)
  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Our approach](#our--approach)
  - [Architecture & overview](#architecture--overview)
  - [Model & stuffs](#model--stuffs)
  - [Conclusion](#conclusion)

 ## Introduction 
    This project is a sligthly modified version of previous ENS Challenge proposed in 2019 by ENEDIS. 

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
│   └── config.ini
├── dataset
│   ├── inout.csv
│   ├── inputs.csv
│   └── outputs.csv
├── docs
│   ├── Makefile
│   ├── _build
│   ├── _static
│   ├── _templates
│   ├── analysis_dataset.html
│   ├── conf.py
│   ├── index.rst
│   └── make.bat
├── jobAM.batch
├── logslurms
│   └── empty.txt
├── notebook
│   ├── notebook_inout.ipynb
│   ├── notebook_inout_dataviz.ipynb
│   ├── notebook_inputs.ipynb
│   ├── notebook_ouputs.ipynb
│   └── prophet_model.ipynb
├── outputs
│   └── test.png
├── requirements.txt
├── src
│   ├── __pycache__
│   ├── configuration.py
│   ├── mailsender.py
│   ├── main.py
│   └── utils.py
└── test
    └── test_merger.py
 ```

 ## Run Model 
```python3 main.py --mode train --model ARIMAX```

 ## Conclusion 
 Since the problem was complicated, we have learned so much about modeling time series and so on. The lecture was just an appetizer to go further on interesting point go get a better score with our model for this project.



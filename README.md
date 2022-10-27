
data
├─ README.md              <-  README for developers using this data-set
└─ public_dataset
|   ├─Peacock_20160101_20171231_15T.csv            <- Peacock site 
|   ├─Rat_20160101_20171231_15T.csv                <- Rat site
|   └─Robin_20160101_20171231_15T.csv              <- Robin site


Peacock_20160101_20171231_15T.csv
Rat_20160101_20171231_15T.csv
Robin_20160101_20171231_15T.csv
Weather and cleaned meter electricity data are organized groupby sites from original datasets below.
In each csv, 10 columns are provided. 1st "dt" column is the timestamp from 2016-01-01 to 2017-12-31 every 15min. 2nd~9th columns are weather features. 10th column is target electricity load.
**************************
Citation of BDG2 Data-Set
Nature Scientific Data (open access)
Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x.
@ARTICLE{Miller2020-yc,
  title     = "The Building Data Genome Project 2, energy meter data from the
               {ASHRAE} Great Energy Predictor {III} competition",
  author    = "Miller, Clayton and Kathirgamanathan, Anjukan and Picchetti,
               Bianca and Arjunan, Pandarasamy and Park, June Young and Nagy,
               Zoltan and Raftery, Paul and Hobson, Brodie W and Shi, Zixiao
               and Meggers, Forrest",
  abstract  = "This paper describes an open data set of 3,053 energy meters
               from 1,636 non-residential buildings with a range of two full
               years (2016 and 2017) at an hourly frequency (17,544
               measurements per meter resulting in approximately 53.6 million
               measurements). These meters were collected from 19 sites across
               North America and Europe, with one or more meters per building
               measuring whole building electrical, heating and cooling water,
               steam, and solar energy as well as water and irrigation meters.
               Part of these data was used in the Great Energy Predictor III
               (GEPIII) competition hosted by the American Society of Heating,
               Refrigeration, and Air-Conditioning Engineers (ASHRAE) in
               October-December 2019. GEPIII was a machine learning competition
               for long-term prediction with an application to measurement and
               verification. This paper describes the process of data
               collection, cleaning, and convergence of time-series meter data,
               the meta-data about the buildings, and complementary weather
               data. This data set can be used for further prediction
               benchmarking and prototyping as well as anomaly detection,
               energy analysis, and building type classification.
               Machine-accessible metadata file describing the reported data:
               https://doi.org/10.6084/m9.figshare.13033847",
  journal   = "Scientific Data",
  publisher = "Nature Publishing Group",
  volume    =  7,
  pages     = "368",
  month     =  oct,
  year      =  2020,
  language  = "en"
}
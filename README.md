# Project Title : Fake Job Posting Prediction

## Project Overview
This project uses data provided from Kaggle. **Data used from:** https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction . The dataset consists of 17,880 observations and 18 features.

This data contains features that define a job posting. These job postings are categorized as either real or fake. Fake job postings are a very small fraction of this dataset. That is as excepted. We do not expect a lot of fake jobs postings. This project follows five stages. The five stages adopted for this project are –

Problem Definition (Project Overview, Project statement and Metrics)
- Data Collection
- Data cleaning, exploring and pre-processing
- Modeling
- Evaluating

This project aims to create a classifier that will have the capability to identify fake and real jobs. The final result will be evaluated based on two different models. Since the data provided has both numeric and text features one model will be used on the text data and the other on numeric data. The final output will be a combination of the two. The final model will take in any relevant job posting data and produce a final result determining whether the job is real or not.

## Algorithms and Techniques

Based on the initial analysis, it is evident that both text and numeric data is to be used for final modeling. Before data modeling a final dataset is determined. This project will use a dataset with these features for the final analysis:

telecommuting
fraudulent
ratio: fake to real job ratio based on location
text: combination of title, location, company_profile, description, requirements, benefits, required_experience, required_education, industry and function
character_count: Count of words in the textual data Word count histogram
Further pre-processing is required before textual data is used for any data modeling.

The algorithms and techniques used in project are:

Natural Language Processing
Naïve Bayes Algorithm
SGD Classifier
Naïve bayes and SGD Classifier are compared on accuracy and F1-scores and a final model is chosen. Naïve Bayes is the baseline model, and it is used because it can compute the conditional probabilities of occurrence of two events based on the probabilities of occurrence of each individual event, encoding those probabilities is extremely useful. A comparative model, SGD Classifier is used since it implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties for classification. This classifier will need high penalties when classified incorrectly. These models are used on both the text and numeric data separately and the final results are combined.

## Benchmark

The benchmark model for this project is Naïve bayes. The overall accuracy of this model is 0.971 and the F1-score is 0.744. The reason behind using this model has been elaborated above. Any other model’s capabilities will be compared to the results of Naïve bayes.

## Methodology

**Data Preprocessing:**
- Tokenization: The textual data is split into smaller units. In this case the data is split into words.
  
- To Lower: The split words are converted to lowercase
  
- Stopword removal: Stopwords are words that do not add much meaning to sentences. For example: the, a, an, he, have etc. These words are removed.
  
- Lemmatization: The process of lemmatization groups in which inflected forms of words are used together.

**Implementation:**
The text dataset is converted into a term-frequency matrix for further analysis. Then using sci-kit learn, the datasets are split into test and train datasets. The baseline model Naïve bayes and another model SGD is trained on the using the train set which is 70% of the dataset. The final outcome of the models based on two test sets – numeric and text are combined such that if both models say that a particular data point is not fraudulent only then a job posting is fraudulent. This is done to reduce the bias of Machine Learning algorithms towards majority classes. The trained model is used on the test set to evaluate model performance. The Accuracy and F1-score of the two models – Naïve bayes and SGD are compared and the final model for our analysis is selected.

**Result:**

The test set has a total of 3265 real jobs and 231 fake jobs. Based on the confusion matrix it is evident that the model identifies real jobs 99.01% of the times. However, fraudulent jobs are identified only 73.5% of the times. Only 2% of the times has the model not identified the class correctly. This shortcoming has been discussed earlier as well as Machine Learning algorithms tend to prefer the dominant classes.


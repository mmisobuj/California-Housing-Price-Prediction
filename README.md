# Machine Learning Assignment



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/miltonraj1148/machine-learning-assignment.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/miltonraj1148/machine-learning-assignment/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)




## Name
Machine Learning Assignment 

1. Md. Mohidul Islam
ID: 20228067

2. Mohammad Atik Ibna Shams
ID: 20228057

3. QUAZI RAIHAN SHAHRIAR
ID: 20228052

4. Md Samiul Huda
ID: 20228038

5. MILTON RAJ BONGHSI
ID: 20228059

WMASDS22: Machine Learning for Data Science

## Dataset

Superstore Marketing Campaign Dataset

https://www.kaggle.com/datasets/ahsan81/superstore-marketing-campaign-dataset


## Objective

We will perform Exploratory Data Analysis, and model the dataset with 3 best performing models. We will also tune their hyper-parameters using GridSearchCV and RandomizedSearchCV.

## Contents

1| Importing necessary libraries, Loading Data and Overview
2| Feature Engineering
3| Exploratory Data Analysis (EDA) Summary
4| Data Preprocessing
5| Model Building
6| Hyperparameter Tuning using GridSearchCV & RandomizedSearchCV
7| Model Performance Comparison and Final Model Selection
8| Feature Importance
9| Pipelines
10| Insights
11| Detailed Exploratory Data Analysis (EDA)

## Data Description

Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise
ID - Unique ID of each customer
YearBirth - Age of the customer Complain - 1 if the customer complained in the last 2 years
DtCustomer - date of customer's enrollment with the company
Education - customer's level of education
Marital - customer's marital status
Kidhome - number of small children in customer's household
Teenhome - number of teenagers in customer's household
Income - customer's yearly household income
MntFishProducts - the amount spent on fish products in the last 2 years
MntMeatProducts - the amount spent on meat products in the last 2 years
MntFruits - the amount spent on fruits products in the last 2 years
MntSweetProducts - amount spent on sweet products in the last 2 years
MntWines - the amount spent on wine products in the last 2 years
MntGoldProds - the amount spent on gold products in the last 2 years
NumDealsPurchases - number of purchases made with discount
NumCatalogPurchases - number of purchases made using catalog (buying goods to be shipped through the mail)
NumStorePurchases - number of purchases made directly in stores
NumWebPurchases - number of purchases made through the company's website
NumWebVisitsMonth - number of visits to company's website in the last month
Recency - number of days since the last purchase

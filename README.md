# DataScienceProj1
This python project is created for training purposes

## ** #homeWork1:
 - task4:
    Need to create the list that contains all the integer numbers from 1
     to 10 using looping and display
     the result on the screen. 
    Solution (for #homeWork1):
    https://github.com/Scout95/DataScienceProj1/blob/master/hw1

## ** #homeWork2: 
 ** 1. task1: 

    Create different modules for the work with data: data_loader.py 
    (for CSV, JSON, API), data_processing.py
     - for handling and 
    transforming data, including some verifications
    -----------------------------------------------
 ** 2. task2: 

    Create methods for histogram, linear graphic, scattered plot. 
    Implement creation of these graphics
    -----------------------------------------------
 ** 3. task3: 

    Create method in module for counting empty or missed values in
     every column of DataFrame and method for 
    printing the report related the missed values
    -----------------------------------------------
    For the home task2 was used the following dataset: 
    
    * Perfume E-Commerce Dataset 2024 *
    https://www.kaggle.com/datasets/kanchana1990/perfume-e-commerce-dataset-2024
    Dataset Overview:
    
    The Perfume E-Commerce Dataset 2024 comprises detailed information on 2000 perfume listings sourced from eBay,
     split into two separate CSV files for men's and women's perfumes, each containing 1000 entries. This dataset 
     provides a comprehensive view of the current market trends, pricing, availability, and geographical distribution
      of perfumes in the e-commerce space.

 ### * Solution (for #homeWork2):*
 https://github.com/Scout95/DataScienceProj1/tree/master/hw2

 ### * Observations and conclusions regarding the plots:

  #### *1 plot (histogram) - Total Sold Quantity of Perfume by Brand
 -----------------------------------------------------------
Calvin Klein is the leader, with a significantly higher sold quantity than any other brand.
Versace also is a strong player, but with a notable gap from Calvin Klein.
There is a steep drop-off after these top two brands, with Elizabeth Taylor and Vera Wang forming a second level.
The distribution is right-skewed: a few brands dominate sales, while many have relatively low volumes.
So, the perfume market is highly concentrated among a few brands.

  #### *2 plot (linear) - Sold Quantity of Perfume by Brand Over Time
 -----------------------------------------------------------
Most brands show sporadic sales activity, with spikes rather than steady trends.
Versace and Calvin Klein show distinct peaks, especially close to the end of the observed period.
Some brands (for example, Vera Wang, Elizabeth Taylor) also have notable spikes, but less consistently.
There are periods of very low or zero sales for many brands, suggesting possible stockouts, seasonality, or reporting gaps.
The most likely, sales are not evenly distributed over time; there are bursts, possibly due to promotions, new launches, 
or seasonal demand.
Monitoring for such spikes can help optimize stock and marketing timing.

  ### *3 plot (scattered) - Sold Quantity Over Time (All Brands)
 ------------------------------------------------------------
Most data points gropped at low sold quantities, with a few extreme outliers.
There is a visible increase in both the number and magnitude of sales towards the end of the period.
The sales process is dominated by frequent small-quantity transactions, but sometimes there are some rare large sales
 (possibly bulk or wholesale orders).
The increase in sales activity towards the end of the period probably, due to the previous plot’s finding—an events or
 season actions.

  ### *4 plot (box plot) - Styled Price Distribution by Brand
 ------------------------------------------------------------

Creed Stands Out as a Premium Brand and has the highest median price and the widest price range among all brands.
Its box is much higher and taller than others, with prices ranging from around $20 up to nearly $275. So, Creed offers
 both entry-level and luxury products, but overall positions itself as a premium brand.

Parfums de Marly, EX NIHILO, and Gwen Stefani Show High Price Variability.These brands have relatively large interquartile ranges (IQRs), indicating significant price diversity within their product lines.

Brands like Lake&skye, TF, Paco Rabanne, Justin Bieber, Xerjoff, Parfum, Giardini Di Toscana, As Picture Show, Vilhelm Parfumerie, Lomani, and Baby Phat display very narrow boxes or even just a line, indicating little to no price variation. These brands offer a single product or products at very similar price points, possibly targeting a specific market segment.

Jo Malone, Valentino, BYREDO have median prices in the $75–$100 range with moderate variability.So, these brands may be targeting the mid-tier market, balancing affordability and luxury.

Creed dominates the high-end segment, with significant price dispersion. This brand could be the focus for luxury marketing and exclusive campaigns.

Brands with narrow price ranges (e.g., Lake&skye, TF, Justin Bieber) may benefit from expanding their lineup to cater to broader customer segments.

Brands with high variability (e.g., Parfums de Marly, EX NIHILO) are likely experimenting with both affordable and premium products, appealing to a wider audience.

There is a clear separation between premium (Creed, Parfums de Marly) and mid-range or budget brands. Brands looking to expand could consider introducing products in the under-served price ranges.
 ---------------------------------------------------------------

## ** #homeWork3: 
** Task: Create a new database and necessary tables, import data from dataset and insert this data into the created table. Prepare and execute SQL requests and the necessary plots.

The following dataset was used for the home work3 task: 
    
   * Perfume E-Commerce Dataset 2024 *
   https://www.kaggle.com/datasets/kanchana1990/perfume-e-commerce-dataset-2024
    
 ### * Solution (for #homeWork3):*
 https://github.com/Scout95/DataScienceProj1/tree/master/hw3


# Below is Perfume Sales Data Analysis - Conclusions and Observations:

## 1. Top 15 Brands by Average Price (Bar Plot)
This bar plot shows the average price of perfumes for the 15 most expensive brands.
 
Brands at the top tend to position themselves as premium with higher price points. This helps identify luxury brands versus more affordable ones. Pricing strategy varies widely; some brands target high-end markets while others compete on affordability.

---

## 2. Top 15 Brands by Total Units Sold (Bar Plot)
This plot displays total units sold aggregated by brand, highlighting the best-selling brands.

Some brands with moderate pricing may achieve higher volume sales, indicating strong market demand or better accessibility.
High sales volume does not always correlate with high price; brands balance price and volume differently.

---

## 3. Top 15 Locations by Number of Sales Records (Bar Plot)
Shows geographic distribution of sales records, identifying key markets.

Certain locations dominate sales activity, which may reflect population density, marketing focus, or regional preferences.
Location plays a critical role in sales performance; targeted regional strategies could optimize revenue.

---

## 4. Scatter Plot of Price vs Units Sold by Perfume Type
Scatter plot illustrating the relationship between price and units sold, colored by perfume type.

Some perfume types may sell well at lower prices, while others maintain sales despite higher prices. Outliers may indicate niche or luxury products.

Price sensitivity varies by perfume type; understanding this helps in inventory and pricing decisions.

---

## 5. Price Distribution by Top 15 Perfume Types (Box Plot)
Box plots showing price ranges and medians for the most common perfume types.

Some types have wider price ranges, indicating diverse product offerings; others are more uniform.
Product variety within types affects pricing strategy and customer choice.

---

## 7. Correlation Heatmap of Numeric Variables
Heatmap showing correlations between numeric variables like price and units sold.

A weak or negative correlation may exist between price and units sold, suggesting higher prices could limit sales volume.
Pricing impacts sales volume but is not the sole factor; other variables like brand and location also influence demand.

---

# Overall Conclusions to optimize pricing, marketing, and inventory decisions in the perfume market

- **Brand Positioning:** Premium brands command higher prices but do not always lead in volume. Balancing price and volume is key.
- **Market Focus:** Concentrated sales in certain locations suggest opportunities for geographic expansion or targeted marketing.
- **Product Mix:** Understanding price sensitivity by perfume type can guide assortment and promotional strategies.
- **Data-Driven Pricing:** Correlation insights support dynamic pricing models tailored by brand, type, and location.

---

## ** #homeWork4: 
 ** 1. Task6: 
** Implement min 5 classificators, compare the metrics, select the best classificator for your dataset.
** Train the model using dataset Credit Card Fraud Detection for fraud operations prediction. Use hyperparametric settings and evaluate results.

The following dataset was used for the home work4 task: 
    
   * Credit Card Fraud Detection *
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
 ### * Solution (for #homeWork4):*
 five classificators: https://github.com/Scout95/DataScienceProj1/tree/master/hw4/fiveClassificators
 catboost results: https://github.com/Scout95/DataScienceProj1/tree/master/hw4


The dataset is highly imbalanced: about 284,315 normal transactions vs. 492 fraudulent ones.
## Model Evaluation and Recommendations

The dataset exhibits extreme class imbalance, with the minority class representing less than 0.2% of samples. This necessitates careful model selection and metric interpretation.

Evaluated five models: CatBoost, AdaBoost, ExtraTrees, QDA, and LightGBM.

**Best Performing Models:**
- **CatBoost** and **ExtraTrees** classifiers demonstrated superior performance, achieving ROC AUC scores above 0.93 and F1-scores around 0.85 on the minority class.
- These models balance precision and recall effectively, making them suitable for fraud detection or anomaly detection tasks where identifying the minority class is critical.

**Metric Considerations:**
- **ROC AUC** is the most reliable metric for evaluating model discrimination on imbalanced data.
- **F1-score (minority class)** is essential to assess the balance between false positives and false negatives.
- **Accuracy** is not informative due to the severe class imbalance and should not be used as the primary performance indicator.

**Models to Avoid:**
- **AdaBoost** — Good but Less Precise, ROC AUC close to CatBoost.
Lower precision (0.74) and recall (0.68), meaning more false alarms and missed frauds.
- **QDA** — High Recall, Very Low Precision, High recall (0.83) means many frauds detected. Very low precision (0.06) indicates many false positives.
- **LightGBM** — Lowest Performance, ROC AUC of 0.86 is significantly lower than others. Low precision (0.31) and moderate recall (0.66).
**QDA** and **LightGBM** showed poor precision and F1-scores on the minority class, despite reasonable recall or accuracy, making them less suitable for this task.

**Summary:**
For imbalanced binary classification problems like this, ensemble methods such as CatBoost and ExtraTrees are recommended. They provide robust detection of the minority class while maintaining high overall accuracy. Evaluation should focus on ROC AUC and minority class F1-score rather than accuracy alone.

Conclusion:
Use **CatBoost** as the main model for fraud detection.

-------------

**Training Model Results:**

1. Data Overview
Dataset size: 284,807 transactions with 31 features.

Class imbalance:

Legitimate (Class 0): 284,315

Fraudulent (Class 1): 492

Fraudulent transactions make up only ~0.17% of the data.

2. Preprocessing
Split: 70% train, 30% test, stratified by class.

Scaling: The Amount feature was standardized.

Balancing:

Random undersampling was applied to the training set, resulting in 344 fraud and 344 non-fraud samples (balanced train set).

Test set remained imbalanced to reflect real-world conditions.

3. Model Selection and Optimization
Algorithm: CatBoost Classifier.

Hyperparameter tuning:

Used Bayesian Optimization to maximize ROC-AUC via 3-fold cross-validation on the balanced training set.

Best found parameters:

depth: 4

iterations: 316

l2_leaf_reg: 10.16

learning_rate: 0.0641

4. Model Training
Trained on the balanced training set with early stopping.

CatBoost’s internal logs show steady improvement in loss and validation metrics.

5. Model Evaluation
Optimal threshold selected to maximize precision while maintaining recall ≥ 0.7.

Test set metrics (on imbalanced data):

ROC-AUC: 0.9730

F1-Score (optimal threshold): 0.7298

Precision (optimal threshold): 0.7591

Recall (optimal threshold): 0.7027

Conclusion
The CatBoost model, optimized with Bayesian hyperparameter search and trained on a balanced dataset, demonstrates strong performance on highly imbalanced real-world data.

High ROC-AUC (0.9730) indicates the model is very good at distinguishing between fraudulent and legitimate transactions.

Precision of 0.76 at Recall 0.70 means that, when the model predicts fraud, it is correct about 76% of the time, while still catching over 70% of all fraud cases.

F1-score of 0.73 shows a good balance between precision and recall at the selected threshold.

Practical implication: The model can be used for real-time fraud detection, prioritizing catching as many fraudulent transactions as possible while keeping false alarms at a manageable level.

---

## ** #homeWork5: 
 ** 1. Task: 
** Implement min 5 regressors, compare the metrics, analyze and select the best regressor for your dataset.

The following dataset was used for the home work5 task: 
    
   * Credit Card Fraud Detection *
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
 ### * Solution (for #homeWork):*
 five classificators: https://github.com/Scout95/DataScienceProj1/tree/master/hw5/

## Comparative Analysis and Results Interpretation

### Summary of Regressor Performance

- **Elastic Net** demonstrated the best results across key metrics (lowest MSE and MAE, highest R²), confirming its ability to effectively model data with regularization, prevent overfitting, and account for feature importance.
- **Ensemble Regressor** also showed high accuracy and stability, only slightly behind Elastic Net, thanks to combining the strengths of several models.
- **Decision Tree** offers good interpretability and average performance but requires careful tuning to avoid overfitting.
- **K-Nearest Neighbors** and **Passive-Aggressive Regressor** performed worst due to sensitivity to dimensionality, noise, and class imbalance.
- **Dummy Regressor** serves as a baseline, showing minimal prediction capability without using any features.

**Conclusion:**  
For this dataset, **Elastic Net** is the optimal choice due to its superior balance of accuracy, robustness. Ensembles and decision trees can be used to improve interpretability and stability but are inferior in prediction quality.

------
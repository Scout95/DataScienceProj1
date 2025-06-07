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

The test set contains 85,443 samples, including 148 fraud cases.

Evaluated five models: CatBoost, AdaBoost, ExtraTrees, QDA, and LightGBM.

Key Metrics Summary
Model	ROC AUC	Precision (Fraud)	Recall (Fraud)	F1-score (Fraud)	Accuracy
CatBoost	0.9695	0.95	0.76	0.84	~1.00
AdaBoost	0.9675	0.74	0.68	0.70	~1.00
ExtraTrees	0.9339	0.97	0.76	0.85	~1.00
QDA	0.9609	0.06	0.83	0.11	0.98
LightGBM	0.8607	0.31	0.66	0.43	~1.00
Detailed Insights
CatBoost — Best Balanced Model

Highest ROC AUC (0.97) and excellent balance of precision (0.95) and recall (0.76).

Accurately detects frauds while minimizing false positives.

Recommended as the primary model for fraud detection.

AdaBoost — Good but Less Precise

ROC AUC close to CatBoost.

Lower precision (0.74) and recall (0.68), meaning more false alarms and missed frauds.

Could be useful in an ensemble or when prioritizing recall.

ExtraTrees — High Precision and Good Recall

Precision of 0.97 and recall of 0.76, close to CatBoost’s performance.

Slightly lower ROC AUC (0.93) but still strong.

Good candidate for ensemble methods.

QDA — High Recall, Very Low Precision

High recall (0.83) means many frauds detected.

Very low precision (0.06) indicates many false positives.

Suitable only if missing fraud is unacceptable and false alarms can be tolerated.

LightGBM — Lowest Performance

ROC AUC of 0.86 is significantly lower than others.

Low precision (0.31) and moderate recall (0.66).

Needs further tuning or data preprocessing.

Conclusion:
Use CatBoost as the main model for fraud detection.

-------------
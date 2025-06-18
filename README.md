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

-------------------------------------------------------------

## ** #homeWork5: 
 ** 1. Task: 
** Implement min 5 regressors, compare the metrics, analyze and select the best regressor for your dataset.

The following dataset was used for the home work5 task: 
    
   * Concrete Compressive Strength Regression *
  https://www.kaggle.com/code/michaelbryantds/concrete-compressive-strength-regression?select=Concrete_Data_Yeh.csv
    
 ### * Solution (for #homeWork5):*
  https://github.com/Scout95/DataScienceProj1/tree/master/hw5/

## Comparative Analysis and Results Interpretation

### Summary of Regressor Performance

## Concrete Compressive Strength Regression

Датасет предназначен для проверки прогноза прочности бетона на сжатие по его составу и возрасту [Concrete_Data_Yeh.csv] (https://www.kaggle.com/code/michaelbryantds/concrete-compressive-strength-regression?select=Concrete_Data_Yeh.csv).

## Анализ выбросов

### Обнаружение и распределение выбросов

Анализ выбросов проводился по методу межквартильного размаха (IQR) для каждого признака. 
 Количество выбросов по каждому признаку:
 cement               0
 slag                 2
 flyash               0
 water                9
 superplasticizer    10
 coarseaggregate      0
 fineaggregate        5
 age                 59
 csMPa                4
 dtype: int64

Наибольшее количество выбросов выявлено в следующих столбцах:

- **age** (59 выбросов) — связано с наличием образцов с очень малым (3 дня) и очень большим (365 дней) возрастом. Это подтвержает статистика по аномальным записям: в консоли выведена статистика по аномалиям, где видно, что среди выбросов по age:
   - Минимальное значение: 3 дня
   - Максимальное значение: 365 дней
   - Среднее значение: 186 дней
   - Стандартное отклонение: 127 дней
   Это подтверждает, что выбросы по age — это либо очень молодые, либо очень "старые" образцы. Причина возникновения: как правило, такие значения появляются, потому что в строительной практике исследуют как "раннюю" прочность (3, 7, 28 дней), так и "долгосрочную" (180, 365 дней), чтобы оценить свойства бетона на разных этапах твердения.

- **superplasticizer** (10), **water** (9), **fineaggregate** (5), **slag** (2).  С чем это может быть связано:
   - superplasticizer (10 выбросов):
   Суперпластификаторы в бетоне применяются в очень разных дозах, иногда для получения сверхпластичных смесей или в экспериментальных рецептурах. Выбросы могут быть связаны с необычно большими или малыми дозировками, которые редко встречаются в стандартных составах.
   - water (9 выбросов):
   Вода затворения может варьироваться в широких пределах. Выбросы могут соответствовать либо очень "жидким" смесям (с высоким водоцементным отношением), либо, наоборот, "жёстким" (очень мало воды).
   - fineaggregate (5 выбросов):
   Мелкий заполнитель (песок) может быть в необычно больших или малых количествах в некоторых рецептах, например, для проверки крайних свойств бетона.
   - slag (2 выброса):
   Шлак как компонент может быть добавлен в очень больших или очень малых количествах в рамках экспериментальных серий.

   Почему могут появляються такие выбросы:
   Это могут быть **экспериментальные задачи**:
   В таких инженерных датасетах часто специально варьируют составы для исследования предельных свойств материала.
   **Технологические особенности**:
   Некоторые значения могут быть связаны с особенностями сырья или условий приготовления смесей.
   **Ошибки измерений**:
   Часть выбросов может появиться из-за ошибок в лабораторных измерениях или записи данных, но в инженерных датасетах чаще всего это именно экспериментальные точки.

- В целевой переменной **csMPa** обнаружено всего 4 выброса, что говорит о том, что экстремальная прочность встречается редко.
- В признаках **cement**, **flyash**, **coarseaggregate** выбросы отсутствуют.

### Статистика по аномальным записям

Статистический анализ аномальных (выбросных) записей показывает, что их средние и стандартные отклонения по признакам, особенно по возрасту и прочности, существенно выше, чем в среднем по датасету. Это подтверждает наличие экстремальных режимов испытаний и составов смесей.

Таким образом, причины выбросов: Экспериментальные и технологические особенности проектирования смесей, исследование крайних режимов, а также возможные лабораторные ошибки.

### Корреляция выбросов с целевой переменной
[ВЛИЯНИЕ ВЫБРОСОВ НА ЦЕЛЕВУЮ ПЕРЕМЕННУЮ]
Корреляция выбросов с целевой переменной:
cement              0.55
slag                0.08
flyash             -0.41
water              -0.16
superplasticizer    0.33
coarseaggregate     0.02
fineaggregate      -0.11
age                -0.05
csMPa               1.00
Name: csMPa, dtype: float64

- Наибольшая положительная корреляция выбросов с целевой переменной наблюдается у **cement** (0.55) и **superplasticizer** (0.33). Экстремальные значения этих компонентов чаще связаны с высокими/низкими значениями прочности.
- Отрицательная корреляция у **flyash** (-0.41) и **water** (-0.16).
Экстремальные значения этих признаков чаще встречаются при пониженной прочности.
- Корреляция выбросов по признаку **age** с прочностью практически отсутствует (-0.05), что говорит о том, что экстремальные значения возраста не всегда приводят к экстремальной прочности.

### Влияние выбросов на целевую переменную

- Выбросы по ключевым компонентам смеси (cement, superplasticizer) могут существенно влиять на прочность бетона.
- Экстремальные значения возраста не всегда приводят к экстремальной прочности.
- Отрицательная корреляция выбросов flyash и water с прочностью указывает, что необычно высокое или низкое содержание этих компонентов связано с пониженной прочностью.

### Корреляционный анализ признаков
Корреляция superplasticizer с другими признаками.

Основные выводы на основании корреляционной матрицы признаков следующие:

Наиболее сильная отрицательная корреляция между water и superplasticizer (-0.66): это технологически обосновано: увеличение суперпластификатора позволяет уменьшить количество воды при сохранении удобоукладываемости (консистенции) смеси.

Положительная корреляция между superplasticizer и flyash (0.38), csMPa (0.37), fineaggregate (0.22):
Это говорит о том, что увеличение суперпластификатора сопровождается увеличением доли золы и некоторым ростом прочности.

Слабые корреляции между большинством других признаков:
Большинство коэффициентов корреляции по модулю меньше 0.4, что указывает на отсутствие сильных линейных зависимостей между компонентами смеси. Отрицательная с age (-0.19), coarseaggregate (-0.27), slag (0.04 — практически отсутствует).

Некоторые технологические закономерности:
Например, cement и csMPa имеют положительную корреляцию, что ожидаемо, так как увеличение цемента обычно повышает прочность.

Тепловая карта корреляций показала, что большинство признаков имеют слабые линейные связи между собой, за исключением отдельных технологических зависимостей (например, отрицательная корреляция между water и superplasticizer, положительная — между cement и csMPa). Это подтверждает, что для качественного прогнозирования прочности бетона необходимы модели, способные выявлять сложные нелинейные зависимости.

Общие выводы по корреляциям: между большинством признаков слабые линейные связи (корреляции по модулю < 0.4), исключение — water-superplasticizer, flyash-superplasticizer, cement-csMPa, что отражает реальные технологические закономерности бетонных смесей. Другими словами, есть технологически обоснованные связи между компонентами смеси, особенно между superplasticizer и water.

### Возможные причины возникновения выбросов

- **Технологические особенности:** Данные содержат результаты экспериментов с экстремальными составами и возрастами бетона, что отражает реальные производственные и исследовательские задачи.
- **Экспериментальные ошибки:** Некоторые выбросы могут быть связаны с ошибками измерения или записи данных.
- **Физико-химические эффекты:** Экстремальные значения компонентов используются для изучения предельных свойств бетона.

### Практические рекомендации

- Для данного датасета выбросы несут важную информацию об экстремальных режимах и не должны быть автоматически удалены.
- Применение **RobustScaler** оправдано, так как он устойчив к выбросам и позволяет моделям корректно работать с такими данными.
- Линейные модели хуже справляются с задачей, в том числе из-за влияния выбросов, при этом ансамблевые методы (RandomForest, LightGBM) показывают высокую устойчивость и точность.

---

### Использованные модели

- DummyRegressor (базовый уровень)
- PassiveAggressiveRegressor
- ElasticNet
- KNeighborsRegressor
- DecisionTreeRegressor
- RandomForestRegressor
- LightGBMRegressor
- VotingRegressor (ансамбль)

### Ансамбль VotingRegressor (описание использованного ансамбля)
 
В работе использован ансамбль VotingRegressor, объединяющий RandomForest, LightGBM и KNeighbors. 

Ансамбль VotingRegressor — это модель, объединяющая предсказания нескольких базовых регрессоров (в данном случае RandomForest, LightGBM и KNeighbors) путём усреднения их выходов. Реализован как:
 voting_regressor = VotingRegressor(
    [
        ("rf", models["RandomForest"]),
        ("lgbm", models["LightGBM"]),
        ("knn", models["KNeighbors"]),
    ]
)
voting_regressor.fit(X_train_scaled, y_train)
y_pred_voting = voting_regressor.predict(X_test_scaled)

Такой ансамбль позволяет:
- повысить устойчивость к выбросам и шуму,
- снизить риск переобучения,
- получить стабильные и качественные предсказания за счёт объединения сильных сторон разных моделей.

VotingRegressor показал одну из лучших метрик качества (R²), уступая только LightGBM, и может быть рекомендован для задач, где важна стабильность результата.

Для анализа вклада каждой базовой модели в ансамбль VotingRegressor были рассчитаны коэффициенты корреляции между предсказаниями каждой модели и итоговым предсказанием ансамбля на тестовой выборке. В ансамбль включены следующие модели:
- RandomForestRegressor
- LightGBMRegressor
- KNeighborsRegressor

**Корреляция между предсказаниями базовых моделей и ансамбля:**

- **RandomForestRegressor:** высокая корреляция с ансамблем (близка к 1), что говорит о значительном влиянии этой модели на итоговое решение VotingRegressor.
- **LightGBMRegressor:** также показывает высокую корреляцию, часто сравнимую с RandomForest, что указывает на сопоставимый вклад в ансамбль.
- **KNeighborsRegressor:** корреляция с ансамблем ниже, чем у деревьев, что свидетельствует о дополнительном разнообразии, которое эта модель вносит в ансамбль, снижая переобучение и повышая устойчивость.

### Визуализация вкладов

На графике вкладов видно, что предсказания RandomForest и LightGBM тесно согласованы с итоговым ансамблем, а KNeighbors иногда отклоняется, что подтверждает его роль как источника дополнительного разнообразия в ансамбле.

### Выводы

- **Основной вклад в итоговое предсказание VotingRegressor вносят RandomForest и LightGBM**, поскольку их предсказания максимально согласованы с результатом ансамбля.
- **KNeighborsRegressor** вносит меньший индивидуальный вклад, но обеспечивает ансамблю дополнительную устойчивость за счет отличающейся структуры ошибок.
- **Ансамбль VotingRegressor выигрывает за счет объединения сильных сторон разных моделей:** деревья (RandomForest, LightGBM) хорошо улавливают сложные закономерности, а KNeighbors добавляет устойчивость к отдельным аномалиям и выбросам.
- Такой подход позволяет повысить стабильность и точность прогнозирования по сравнению с использованием одной модели, особенно на сложных и шумных данных.


## Анализ сгенерированных признаков и их ранжирования

### Корреляция выбросов с целевой переменной

При анализе корреляции выбросов по каждому признаку с целевой переменной (`csMPa`) были выявлены следующие важные моменты:

- **Наиболее сильную положительную корреляцию с выбросами прочности показывают:**
  - `cement` (0.73)
  - `cement_to_water` (0.71)
  - `binder` (0.57)
  - `age` (0.33)
  - `superplasticizer` (0.22)

- **Сильная отрицательная корреляция:**
  - `slag_to_cement` (-0.49)
  - `slag` (-0.34)
  - `fineaggregate` (-0.24)
  - `flyash` (-0.25)
  - `water` (-0.18)
  - `fine_to_coarse` (-0.18)
  - `total_agg` (-0.17)

**Вывод:**  
Сгенерированные признаки, такие как `cement_to_water` (водоцементное отношение), `slag_to_cement` и `binder`, обладают высокой корреляцией с выбросами по прочности, что подтверждает их информативность и инженерную значимость. Положительная корреляция указывает на то, что экстремальные значения этих признаков часто сопровождаются экстремальными значениями прочности бетона.

---

### Корреляция superplasticizer с другими признаками

- `superplasticizer_to_cement` (0.84), `binder` (0.40), `flyash` (0.38), `csMPa` (0.37), `fine_to_coarse` (0.33), `cement_to_water` (0.33) — все эти признаки положительно коррелируют с superplasticizer.
- Самая сильная отрицательная корреляция у `water` (-0.66), что технологически объяснимо: увеличение суперпластификатора позволяет уменьшить воду.

**Вывод:**  
Сгенерированные соотношения, такие как `superplasticizer_to_cement`, оказываются тесно связаны с исходным признаком superplasticizer, а также с целевой переменной. Это подтверждает, что такие инженерные коэффициенты улавливают важные зависимости в данных.

---

### Ранжирование признаков по важности (RandomForest)

Топ-5 признаков по важности для RandomForestRegressor:

| Признак                  | Важность |
|--------------------------|----------|
| binder                   | 0.34     |
| age                      | 0.34     |
| cement_to_water          | 0.15     |
| water                    | 0.03     |
| cement                   | 0.03     |

- **binder** (суммарное количество вяжущих) и **age** (возраст) — два самых важных признака, что подтверждает их ключевую роль в формировании прочности.
- **cement_to_water** (водоцементное отношение) — также в топе, что полностью согласуется с инженерными представлениями о бетоне.
- Остальные сгенерированные признаки, такие как `slag_to_cement`, `superplasticizer_to_cement`, `fine_to_coarse`, имеют меньший, но не нулевой вклад.

Таким образом, ключевыми оказались инженерные признаки (binder, cement_to_water) и возраст бетона, что подтверждено анализом важности в лучших моделях.
---


### Анализ подобранных гиперпараметров моделей ###
Описание
В ходе работы были проведены подбор и анализ гиперпараметров для ключевых регрессионных моделей: RandomForestRegressor, KNeighborsRegressor, ElasticNet и LightGBM. Подбор осуществлялся с помощью кросс-валидации и методов GridSearchCV/RandomizedSearchCV, что позволило повысить качество предсказаний на тестовой выборке.

Итоговые параметры моделей:
 --------------------------------------------------------------------
Модель               |	Лучшие гиперпараметры   |	R² (CV) | R² (Test)
RandomForestRegressor|	max_depth=20,           |          |
                     | min_samples_split=2,     |          |
                     | n_estimators=100	      |  0.906	  | 0.894
KNeighborsRegressor	| n_neighbors=7,           |          |
                     | weights='distance'	      |  0.768	  | 0.807
ElasticNet	         | alpha=0.01, l1_ratio=0.1	|  0.595	  | 0.625
LightGBM	            | num_leaves=31,           |          |
                     | n_estimators=200,        |          |
                     | min_data_in_leaf=10,     |          |
                     | learning_rate=0.1	      |  0.917	  | 0.921
 ---------------------------------------------------------------------                  
Краткий анализ гиперпараметров:
RandomForestRegressor
max_depth=20: Позволяет деревьям быть достаточно глубокими для выявления сложных паттернов, но ограничивает переобучение.

min_samples_split=2: Минимальное ограничение, что делает модель чувствительной к локальным закономерностям.

n_estimators=100: Достаточное количество деревьев для стабильности ансамбля.

Вывод: Модель хорошо справляется с задачей, но чуть уступает LightGBM по качеству на тесте.

KNeighborsRegressor
n_neighbors=7: Оптимальный баланс между локальной и глобальной аппроксимацией.

weights='distance': Ближайшие точки оказывают больший вклад, что снижает влияние шумов.

Вывод: Модель чувствительна к локальным выбросам, но показывает достойный результат среди простых моделей.

ElasticNet
alpha=0.01: Слабая регуляризация, что говорит о необходимости учитывать большинство признаков.

l1_ratio=0.1: Преимущественно L2-регуляризация, что характерно для задач с коррелирующими признаками.

Вывод: Линейная модель ограничена по качеству, но важна для сравнения с более сложными подходами.

LightGBM
num_leaves=31, n_estimators=200: Позволяет выявлять сложные зависимости и обеспечивает высокую гибкость.

min_data_in_leaf=10: Снижает риск переобучения на редких паттернах.

learning_rate=0.1: Оптимальный темп обучения для баланса между скоростью и качеством.

Вывод: Лучшая модель по всем метрикам, особенно эффективна на инженерных признаках.

Влияние подбора гиперпараметров
Автоматизированный подбор позволил каждой модели максимально раскрыть потенциал на конкретных данных.

Ансамблевые методы (LightGBM, RandomForest, VotingRegressor) существенно превосходят по качеству простые и линейные модели.



### Оценка качества

Модели сравнивались по метрикам R², MSE и MAE. 
Краткий обзор результатов

**VotingRegressor (ансамбль)** показал очень хорошие результаты:

R² = 0.903 — высокая доля объяснённой дисперсии.

MSE = 25.123, MAE = 3.511 — хорошие показатели ошибки.

Лучшей по R² оказалась **LightGBM**:

R² = 0.921 (лучше, чем у ансамбля).

MSE = 20.446, MAE = 2.922 — лучшие значения ошибок.

Другие модели (**RandomForest, DecisionTree, KNeighbors**) уступают LightGBM и ансамблю по качеству.

Сравнение моделей
 -------------------------------------------------------
Модель	        |  R²	 |  MSE	|  MAE
LightGBM	        | 0.921|	20.446| 2.922
VotingRegressor  | 0.903|	25.123| 3.511
RandomForest	  | 0.894|	27.424| 3.460
DecisionTree	  | 0.827|	44.576| 4.207
KNeighbors	     | 0.807|	49.778| 5.319
ElasticNet	     | 0.625|	96.614| 7.781
PassiveAggressive|0.479 | 134.127| 9.416
Dummy	           | ~0	| 257.717| 13.052

**LightGBM** — лидер по всем метрикам.

**VotingRegresso**r (ансамбль из нескольких моделей) улучшает результаты по сравнению с большинством базовых моделей, но уступает LightGBM.

**RandomForest** близок к ансамблю, но немного хуже.

Анализ вклада базовых моделей в ансамбль
Корреляция предсказаний ансамбля с LightGBM — очень высокая (0.990), что говорит о доминирующем влиянии LightGBM в ансамбле.

Корреляция с RandomForest (0.335) и KNeighbors (0.220) гораздо ниже.

Это объясняет, почему ансамбль не превосходит LightGBM — ансамбль в основном повторяет предсказания **LightGBM**.

Важность признаков в базовых моделях
RandomForest:

Ведущие признаки: binder, age, cement_to_water (около 0.34, 0.34, 0.15).

Остальные признаки имеют гораздо меньший вклад.

LightGBM:

Ведущие признаки по убыванию важности: age, water, coarseaggregate, cement_to_water, binder.

Значения важности представлены в условных единицах, но показывают схожий набор ключевых признаков с RandomForest.

### Выводы:

**LightGBM** — лучшая модель для текущей задачи.
Если цель — максимальное качество, стоит использовать именно её.

**Ансамбль VotingRegressor** не улучшил результат LightGBM, т.к. LightGBM доминирует в ансамбле.
Визуализация предсказаний LightGBM выполнена.
Ошибки модели (MSE, MAE) достаточно низкие, что говорит о хорошей способности моделей к обобщению.

**LightGBMRegressor** рекомендован как лучший выбор для задачи предсказания прочности бетона на сжатие: он обеспечивает максимальное качество (R² ≈ 0.921) и минимальные ошибки. 

------------------------------------------------------------

## ** #homeWork6: 
 ** Финальная работа
1.	Взять набор данных исходя из ваших интересов.
2.	Не используйте датасеты, которые вы уже брали.
3.	Описать колонки, какие характеристики.
4.	Проведите анализ EDA.
5.	Провести предварительную обработку данных, если это необходимо (сделать данные понятными для модели машинного обучения: заполнить пропущенные значения, заменить категориальные признаки и т.д.)
6.	Решить задачу сегментации или анализа временного ряда при помощи не менее 5-ти подходов ML. Составьте ансамбль моделей.
7.	Решить задачу поиска аномалий.
8.	Визуализация. Создать графики ошибок прогнозирования, метрик качества обученной модели и важности признаков.
9.	Результат выполнения финальной работы разместить в гит репозиторий.


The following dataset was used for the home work6 task: 
**Customer Shopping Dataset**
  https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset
    
 ### * Solution (for #homeWork6):*
  https://github.com/Scout95/DataScienceProj1/tree/master/hw6/

## About this Dataset
This dataset contains shopping information from 10 different shopping malls between 2021 and 2023. There is gathered data from various age groups and genders to provide a comprehensive view of shopping habits in Istanbul. The dataset includes essential information such as invoice numbers, customer IDs, age, gender, payment methods, product categories, quantity, price, order dates, and shopping mall locations.

## Attribute Information:

- invoice_no: Invoice number. Nominal. A combination of the letter 'I' and a 6-digit integer uniquely assigned to each operation.
- customer_id: Customer number. Nominal. A combination of the letter 'C' and a 6-digit integer uniquely assigned to each operation.
- gender: String variable of the customer's gender.
- age: Positive Integer variable of the customers age.
- category: String variable of the category of the purchased product.
- quantity: The quantities of each product (item) per transaction. Numeric.
- price: Unit price. Numeric. Product price per unit in Turkish Liras (TL).
- payment_method: String variable of the payment method (cash, credit card or debit card) used for the transaction.
- invoice_date: Invoice date. The day when a transaction was generated.
- shopping_mall: String variable of the name of the shopping mall where the transaction was made.

** В работе при использовании датасета в процессе всех поиска пропусков и бесконечных значений в признаках, используемых для моделирования, категориальные и числовые признаки были закодированы и преобразованы (например, gender_enc, total_purchase_amount, dummy-переменные для категорий и способов оплаты). Диапазоны признаков: возраст — от 18 до 69, total_purchase_amount — от 5.23 до 26 250, количество — от 1 до 5. Бинарные признаки (категории, способы оплаты, торговые центры, месяц, день недели) принимают значения True/False.

## В работе используются методы машинного обучения для сегментации данных:

**Gaussian Mixture Model (GMM)**: Это вероятностный алгоритм кластеризации, который относится к методам машинного обучения без учителя. Он сегментирует данные, определяя принадлежность объектов к кластерам на основе вероятностей.

**Consensus Clustering (Ensemble Clustering)**: Используется функция cluster_ensembles из библиотеки scikit-learn-extra, которая объединяет результаты нескольких кластеризаций для получения более устойчивого и согласованного разбиения данных.

## Реализовано также:
1. **Soft Voting/Probabilistic Assignment via GMM**

Использует обученную модель ** Gaussian Mixture Model (GMM)** для получения вероятностей принадлежности каждого объекта к каждому кластеру (gmm.predict_proba(X_scaled)).

Для каждого объекта выбирается кластер с максимальной вероятностью (soft voting), и эти метки сохраняются в столбец ensemble_cluster_soft_gmm датафрейма.

Выводится распределение объектов по кластерам (сколько объектов попало в каждый кластер).

Строится scatterplot (UMAP-пространство), где цветом отмечены кластеры, определённые soft voting GMM.

Если найдено более одного кластера, вычисляется silhouette score — метрика качества кластеризации (чем выше, тем лучше разделены кластеры).

Результаты:

Получаю распределение по кластерам на основе вероятностного подхода GMM.

Визуализация кластеров в UMAP-пространстве позволяет оценить их разделимость.

Silhouette score показывает, насколько хорошо объекты разделены между кластерами.

2. **Consensus Functions** from sklearn-extra

Использую библиотеку scikit-learn-extra и функцию cluster_ensembles для объединения результатов разных кластеризаций (ensemble clustering).

Поддерживаются методы CSPA, HGPA, MCLA (в данном случае — CSPA).

На входе - матрица меток кластеров от разных моделей (cluster_labels_matrix), на выходе — итоговые метки кластеров (consensus_labels), которые сохраняются в столбец ensemble_cluster_sklearn_extra.

Выводится распределение по кластерам после консенсус-кластеризации.

Строится scatterplot (UMAP-пространство) с цветовой маркировкой по итоговым меткам кластеров.

Если найдено более одного кластера, вычисляется silhouette score для оценки качества консенсус-кластеризации.

## Результаты:

Получаю итоговые метки кластеров, объединяющие результаты разных алгоритмов кластеризации.

Визуализация и silhouette score позволяют сравнить качество консенсус-кластеризации с другими подходами.

Итоговые выводы
Код реализует два подхода ансамблирования кластеризации: soft voting через вероятности GMM и consensus clustering через sklearn-extra.

Оба подхода сохраняют метки кластеров в датафрейме, строят визуализации и оценивают качество кластеризации с помощью silhouette score.

Soft voting GMM позволяет учитывать неопределённость в принадлежности кластерам, а consensus clustering агрегирует результаты нескольких моделей для более устойчивого решения.

## Реализованы ансамблевые подходы:

Ансамбль кластеризаций: С помощью функции cluster_ensembles объединяются метки кластеров, полученные разными алгоритмами или разными запусками одного алгоритма. Это позволяет повысить устойчивость и качество итоговой сегментации.

Soft Voting (GMM): Хотя soft voting в данном случае реализован на основе одной модели GMM, он использует вероятностное распределение по кластерам, что позволяет учитывать неопределённость в принадлежности объектов к кластерам.

## Метрика качества кластеризации — silhouette score, 
вычисляется и выводится а также визуализируется распределение кластеров на двумерном пространстве (UMAP), что позволяет оценить качество разбиения визуально.

## Выводы по результатам сегментации и аномалий
Код успешно реализовал сегментацию клиентов методом кластеризации (KMeans + UMAP), разделив данные на 5 кластеров. Это видно по таблице средних значений признаков по каждому кластеру, что позволяет интерпретировать и сравнивать группы по возрасту, полу, категориям покупок, способам оплаты, времени и месту совершения покупок. Такой подход помогает выявить сегменты с разными поведенческими и демографическими характеристиками.

Качество кластеризации оценивается метрикой silhouette score (0.325) — это говорит о среднем качестве разделения: кластеры различимы, но есть некоторое пересечение между ними.

Проведена проверка на аномалии тремя методами (Isolation Forest, LOF, One-Class SVM): количество аномалий близко для всех методов (~995–1064). Это позволяет выявить клиентов с нетипичным поведением, что может быть полезно для дальнейшего анализа или для исключения выбросов при построении моделей.

В данных нет пропусков (NaN) и дубликатов (уникальных строк больше 79 тысяч при общем числе ~99 тысяч), что говорит о хорошем качестве исходного массива.

В результате сегментации получены усреднённые профили клиентов по каждому кластеру. Например, можно видеть различия по среднему возрасту, количеству и сумме покупок, а также по вероятности покупки определённых категорий товаров или посещения конкретных торговых центров. Это важно для маркетинга, персонализации предложений и построения портретов целевых групп.

Результаты и профили кластеров сохранены для дальнейшего анализа. Это позволяет использовать сегментацию для практических бизнес-задач: таргетинга, персонализации, выявления новых рыночных ниш.

## В работе осуществлялся запуск и обучение модели на TensorFlow
Обучение модели происходит в течение 50 эпох (Epoch 1/50 ... Epoch 50/50).

Для каждой эпохи показаны:

Количество шагов (например, 2798/2798).

Время на эпоху (около 3-5 секунд).

Значения функции потерь (loss) на обучающей выборке и на валидационной (val_loss).

Значения loss и val_loss постепенно уменьшаются, что свидетельствует об успешном обучении модели и улучшении качества.

После обучения модели происходит кластеризация (Deep Learning кластеризация), распределение по 5 кластерам (deep_cluster с номерами 0–4).

Размеры кластеров:

Кластер 1 — 56 621 объектов (самый крупный)

Кластер 2 — 14 618

Кластер 4 — 10 542

Кластер 0 — 9 779

Кластер 3 — 7 897

Вывод повторяется дважды — возможно, дублирование вывода.

Оценка качества кластеризации

Silhouette Score для UMAP + KMeans: 0.324 — умеренно хороший показатель, указывающий на достаточно четкие кластеры.

Silhouette Scores для других методов:

KMeans: -0.017 (плохой, отрицательный)

GMM: 0.060 (низкий)

DBSCAN: 0.326 (лучший среди перечисленных, близок к UMAP+KMeans)

Аномалии по разным методам

Количество обнаруженных аномалий:

Isolation Forest: 995

Local Outlier Factor (LOF): 995

One-Class SVM: 1064

Это говорит о том, что в данных есть порядка 1000 аномальных наблюдений, выявленных разными методами.

Статистика по кластерам

Таблица с усредненными значениями признаков по кластерам (возраст, закодированный пол, дни недели покупок и др.).

Это позволяет понять характеристики каждого сегмента клиентов.

Сохранение результатов

Результаты сегментации и аномалий сохранены в файл customer_shopping_segmented.csv.

EDA и признаки

Выведен список колонок итогового датафрейма, включая признаки, кластеры и аномалии.

Указана важность признаков для предсказания кластера (пример RandomForest на KMeans).

Consensus Clustering (ансамблевая кластеризация)

Распределение по кластерам на основе большинства голосов разных алгоритмов:

Кластер 1 — 48 809 объектов

Кластер 0 — 21 889

Кластер 3 — 13 693

Кластер 2 — 11 844

Кластер 4 — 3 222

## Итоговые выводы
Модель успешно обучена: функция потерь на обучении и валидации стабильно снижается, что говорит о хорошей сходимости.

Кластеризация выявила 5 сегментов клиентов, с достаточно четкими границами (silhouette score ~0.32).

Аномалии выявлены разными методами, их количество примерно 1000, что может помочь в дальнейшем анализе или очистке данных.

Результаты сегментации и аномалий сохранены для дальнейшего использования.

Ансамблевая кластеризация позволяет получить более устойчивое распределение по кластерам.

## EDA и важность признаков 
дают понимание, какие факторы влияют на сегментацию клиентов.

На основе данных и контекста сегментации клиентов, можно выделить следующие факторы, влияющие на сегментацию клиентов:

** Влияющие факторы на сегментацию клиентов и их числовые показатели**
- Возраст (age)
В таблице со статистикой по кластерам средний возраст варьируется примерно от 43.2 до 43.7 лет по разным сегментам, что указывает на некоторую дифференциацию по возрасту между кластерами.

- Закодированный пол (gender_enc)
Значения варьируются от 0.37 до 0.47, что отражает различия в распределении полов по сегментам (например, больше мужчин или женщин в разных кластерах).

- Категориальные признаки покупок (category_Clothing, category_Cosmetics и др.)
Важность категорий отражается в бинарных признаках (True/False), которые участвуют в сегментации. Например, доля покупок в категориях одежды, косметики и т.п. различается по кластерам, что влияет на формирование сегментов.

- Метод оплаты (payment_method_Credit Card, Debit Card и др.)
Присутствие бинарных признаков методов оплаты указывает на то, что способ оплаты — значимый фактор сегментации.

- Место покупки (shopping_mall_*)
Различия по торговым центрам (например, Emaar Square Mall, Forum Istanbul и др.) также важны для разделения клиентов на группы.

- Время покупки (purchase_month_, purchase_dayofweek_)
Признаки месяца и дня недели покупки влияют на сегментацию, что видно из распределения по дням недели в таблице.

- Итоговая сумма покупки (total_purchase_amount)
Значения суммы покупок варьируются от минимальных до максимальных (например, от 5.23 до 26250), что является важным дифференцирующим признаком.

- Аномалии (iso_anomaly, lof_anomaly, ocsvm_anomaly)
Наличие аномальных наблюдений в данных (около 1000 по разным методам) также влияет на сегментацию, выделяя особые группы клиентов.
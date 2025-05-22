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



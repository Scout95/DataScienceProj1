import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data_from_db(db_path="datascience1.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT brand, title, type, price, sold, date_of_sale, location_of_sale 
        FROM top_perfume_brands
    """,
        conn,
    )
    df["date_of_sale"] = pd.to_datetime(df["date_of_sale"], errors="coerce")
    conn.close()
    return df


def plot_average_price_by_brand(df):
    avg_price = (
        df.groupby("brand")["price"].mean().sort_values(ascending=False).head(15)
    )
    plt.figure(figsize=(12, 6))
    # sns.barplot(x=avg_price.values, y=avg_price.index, palette="viridis")
    sns.barplot(x=avg_price.values, y=avg_price.index, color='steelblue')
    plt.title("Top 15 Brands by Average Price")
    plt.xlabel("Average Price")
    plt.ylabel("Brand")
    plt.tight_layout()
    # plt.show()


def plot_total_units_sold_by_brand(df):
    total_sold = df.groupby("brand")["sold"].sum().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    # sns.barplot(x=total_sold.values, y=total_sold.index, palette="rocket")
    sns.barplot(x=total_sold.values, y=total_sold.index, color='steelblue')
    plt.title("Top 15 Brands by Total Units Sold")
    plt.xlabel("Total Units Sold")
    plt.ylabel("Brand")
    plt.tight_layout()
    # plt.show()


def plot_sales_distribution_by_location(df):
    location_counts = df["location_of_sale"].fillna("N/A").value_counts().head(15)
    plt.figure(figsize=(12, 6))
    # sns.barplot(x=location_counts.values, y=location_counts.index, palette="mako")
    sns.barplot(x=location_counts.values, y=location_counts.index, color='steelblue')
    plt.title("Top 15 Locations by Number of Sales Records")
    plt.xlabel("Number of Sales Records")
    plt.ylabel("Location")
    plt.tight_layout()
    # plt.show()


def plot_price_vs_sold_scatter_top15(df):
    top_types = df['type'].value_counts().nlargest(15).index.tolist()
    df_top15 = df[df['type'].isin(top_types)]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_top15, x='price', y='sold', hue='type', alpha=0.7)
    # sns.scatterplot(data=df, x="price", y="sold", hue="type", alpha=0.7)
    plt.title("Scatter Plot of Price vs Units Sold by Perfume Type")
    plt.xlabel("Price")
    plt.ylabel("Units Sold")
    # plt.legend(title="Type")
    plt.legend(
        title="Type",
        loc="center left",  # Position the legend's anchor point
        bbox_to_anchor=(
            1,
            0.5,
        ),  # Place legend just outside the plot on the right center
        fontsize="small",  # Make legend font smaller
        title_fontsize="medium",  # Legend title font size
        borderaxespad=0,  # Padding between legend and axes
    )
    plt.tight_layout()
    # plt.show()


# def plot_monthly_sales_trend(df):
#     df["sale_month"] = df["date_of_sale"].dt.to_period("M")
#     monthly_sales = df.groupby("sale_month")["sold"].sum().reset_index()
#     monthly_sales["sale_month"] = monthly_sales["sale_month"].dt.to_timestamp()
#     plt.figure(figsize=(14, 6))
#     sns.lineplot(data=monthly_sales, x="sale_month", y="sold")
#     plt.title("Monthly Total Units Sold Over Time")
#     plt.xlabel("Month")
#     plt.ylabel("Total Units Sold")
#     plt.tight_layout()
#     plt.show()


def plot_price_distribution_by_type(df):
    top_types = df['type'].value_counts().nlargest(15).index.tolist()
    df_top15 = df[df['type'].isin(top_types)]
    
    plt.figure(figsize=(12,6))
    sns.boxplot(x='type', y='price', data=df_top15)
    plt.title("Price Distribution by Top 15 Perfume Types")
    plt.xlabel("Perfume Type")
    plt.ylabel("Price")
    plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
    plt.tight_layout()
    # plt.show()


def plot_correlation_heatmap(df):
    numeric_df = df[["price", "sold"]].copy()
    corr = numeric_df.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Price and Sold")
    plt.tight_layout()
    # plt.show()


def run_all_plots(db_path="datascience1.db"):
    df = load_data_from_db(db_path)
    plot_average_price_by_brand(df)
    plot_total_units_sold_by_brand(df)
    plot_sales_distribution_by_location(df)
    plot_price_vs_sold_scatter_top15(df)
    # plot_monthly_sales_trend(df)
    plot_price_distribution_by_type(df)
    plot_correlation_heatmap(df)
    plt.show()
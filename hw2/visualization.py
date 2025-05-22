import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from io import StringIO


class VisualizationModule:
    def __init__(self, figsize=(12, 6), style="seaborn", facecolor="white", title=None):
        self.fig, self.ax = plt.subplots(figsize=figsize)  # Set figsize here
        self.plots = {}

    def add_histogram(self, data, bins=10, label=None):
        hist = self.ax.hist(data, bins=bins, label=label)
        self.plots[label] = hist
        if label:
            self.ax.legend()
        plt.show()

    def remove_histogram(self, label):
        if label in self.plots:
            for patch in self.plots[label]:
                patch.remove()
            del self.plots[label]
            self.ax.legend()
            plt.draw()

    def add_line_plot(self, x, y, label=None):
        (line,) = self.ax.plot(x, y, label=label)
        self.plots[label] = line
        if label:
            self.ax.legend()
        plt.show()

    def remove_line_plot(self, label):
        if label in self.plots:
            self.plots[label].remove()
            del self.plots[label]
            self.ax.legend()
            plt.draw()

    def add_scatter_plot(self, x, y, label=None):
        scatter = self.ax.scatter(x, y, label=label)
        self.plots[label] = scatter
        if label:
            self.ax.legend()
            plt.show()

    def remove_scatter_plot(self, label):
        if label in self.plots:
            self.plots[label].remove()
            del self.plots[label]
            self.ax.legend()
            plt.draw()

    # def plot_quantity_by(self, csv_data: str, groupBy: str, columnName: str, titleName: str, xlabelName: str, yLabelName: str):
    def plot_quantity_by(
        self,
        csv_data: str,
        groupBy: str,
        columnName: str,
        titleName: str,
        xlabelName: str,
        yLabelName: str,
    ):
        # processed_df = pd.read_csv(StringIO(csv_data))
        # grouped_data = processed_df.groupby(groupBy)[columnName].sum().sort_values(ascending=False)
        grouped_data = (
            csv_data.groupby(groupBy)[columnName].sum().sort_values(ascending=False)
        )
        #  plt.figure(figsize=(15, 6))
        grouped_data.plot(kind="bar", color="skyblue")
        plt.title(titleName)
        plt.xlabel(xlabelName)
        plt.ylabel(yLabelName)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_sold_quantity_by_limit(
        self,
        df: pd.DataFrame,
        groupBy: str,
        columnName: str,
        titleName: str,
        xlabelName: str,
        yLabelName: str,
        top_n: int = 20,
    ):
        # Group, sum, and select top N
        sold_by_brand = df.groupby(groupBy)[columnName].sum().nlargest(top_n)
        # Plotting
        # plt.figure(figsize=(12, 8))
        sold_by_brand.plot(kind="bar", color="skyblue")
        plt.title(titleName)
        plt.xlabel(xlabelName)
        plt.ylabel(yLabelName)
        plt.xticks(rotation=45)
        plt.tight_layout()

    # plt.show()

    def plot_sold_quantity_over_time(
        self, perfume_df, title, xlabelTitle, yLabelTitle, legendTitle
    ):
        """
        Method plots the sold quantity of perfume by brand over time.
        How it works:
        - Converts the 'lastUpdated' column to datetime format, coercing errors to NaT.
        - Groups the data by 'brand' and by the date part of 'lastUpdated'.
        - Sums the 'sold' quantities for each brand-date group.
        - Transposes the grouped data so dates are on the x-axis and brands are separate lines.
        - Plots the data as a line chart with markers.
        - Adds the provided titles and labels for clarity.
        Parameters:
        - perfume_df: pandas DataFrame containing at least 'brand', 'lastUpdated', and 'sold' columns.
        - title: Title of the plot.
        - xlabelTitle: Label for the x-axis.
        - yLabelTitle: Label for the y-axis.
        - legendTitle: Title for the legend.
        Usage example:
        >>> plot_sold_quantity_over_time(
        ...     perfume_df,
        ...     title="Sold Quantity of Perfume by Brand Over Time",
        ...     xlabelTitle="Date",
        ...     yLabelTitle="Sold Quantity",
        ...     legendTitle="Brand"
        ... )
        """
        # --- Limit to top 20 brands by total sold quantity ---
        top_brands = (
            perfume_df.groupby("brand")["sold"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .index
        )
        perfume_df = perfume_df.copy()
        perfume_df = perfume_df[perfume_df["brand"].isin(top_brands)]

        # # Convert 'lastUpdated' to datetime, invalid parsing will be NaT
        # perfume_df["lastUpdated"] = pd.to_datetime(
        #     perfume_df["lastUpdated"], errors="coerce"
        # )
        # Remove the timezone abbreviation (e.g. "PDT") from the datetime strings
        perfume_df["lastUpdated_stripped"] = perfume_df["lastUpdated"].str.replace(
            r"\sPDT$", "", regex=True
        )
        # Convert the cleaned strings to datetime without timezone info
        perfume_df["lastUpdated"] = pd.to_datetime(
            perfume_df["lastUpdated_stripped"], errors="coerce"
        )
        # Localize the naive datetime to the correct timezone (e.g., US/Pacific)
        perfume_df["lastUpdated"] = perfume_df["lastUpdated"].dt.tz_localize(
            "US/Pacific"
        )
        # Drop the temporary stripped column if no longer needed
        perfume_df.drop(columns=["lastUpdated_stripped"], inplace=True)

        # Group by brand and date (date part only), sum sold quantities
        sold_by_brand_date = (
            perfume_df.groupby(["brand", perfume_df["lastUpdated"].dt.date])["sold"]
            .sum()
            .unstack(level=0)
        )

        sold_by_brand_date.plot(marker="o", ax=self.ax)  # Use self.ax
        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel(xlabelTitle, fontsize=10)
        self.ax.set_ylabel(yLabelTitle, fontsize=10)
        self.ax.tick_params(axis="x", rotation=45, labelsize=10)
        self.ax.tick_params(axis="y", labelsize=10)
        self.ax.legend(
            title=legendTitle, fontsize=9, title_fontsize=10, loc="upper left"
        )
        self.fig.tight_layout()  # Use self.fig

    def plot_scatter_sold_by_brand_over_time(
        self, perfume_df, title, xlabelTitle, yLabelTitle, legendTitle
    ):
        """
        Creates a scatter plot of sold quantity by brand over time from CSV data.
            - Converts the 'lastUpdated' column to datetime, coercing errors to NaT.
            - Iterates over each unique brand and plots sold quantity vs. date as scatter points.
            - Adds plot title, axis labels, legend with specified legend title, and formats x-axis dates.
            - Displays the plot.
            Parameters:
            - csv_data (str): CSV formatted string containing at least 'brand', 'lastUpdated', and 'sold' columns.
            - title (str): Plot title.
            - xlabelTitle (str): Label for x-axis.
            - yLabelTitle (str): Label for y-axis.
            - legendTitle (str): Title for the legend.
            Example:
            >>> plot_scatter_sold_by_brand_over_time(csv_data,
            ...     title="Scatter Plot of Sold Quantity by Brand Over Time",
            ...     xlabelTitle="Date",
            ...     yLabelTitle="Sold Quantity",
            ...     legendTitle="Brand")
        """
        # Ensure datetime column is parsed and timezone-naive
        perfume_df = perfume_df.copy()
        perfume_df.loc[:, "lastUpdated"] = perfume_df["lastUpdated"].str.replace(
            "-", ""
        )  #

        perfume_df.loc[:, "lastUpdated"] = perfume_df["lastUpdated"].str.replace(
            r"\sPDT$", "", regex=True
        )

        perfume_df["lastUpdated"] = pd.to_datetime(
            perfume_df["lastUpdated"], errors="coerce"
        )
        perfume_df = perfume_df.dropna(subset=["lastUpdated", "sold"])
        perfume_df["lastUpdated_naive"] = perfume_df["lastUpdated"].dt.tz_localize(None)

        #     plt.figure(figsize=(10, 7))
        plt.scatter(perfume_df["lastUpdated_naive"], perfume_df["sold"])
        plt.xlabel("Date")
        plt.ylabel("Sold Quantity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()

    def styled_brand_boxplot(
        self,
        perfume_df,
        price_col="price",
        brand_col="brand",
        title="Styled Price Distribution by Brand",
        n_brands=20,
    ):
        """
        Creates a styled boxplot of price distribution by brand.
        Args:
            perfume_df (pd.DataFrame): DataFrame containing the data.
            price_col (str): Name of the column with price data.
            brand_col (str): Name of the column with brand data.
            title (str): Title for the plot.
        """
        

        # top_brands = perfume_df[brand_col].value_counts().head(n_brands).index
        top_brands = perfume_df.groupby(brand_col)[price_col].median().sort_values(ascending=False).head(n_brands).index

        df_filtered = perfume_df[perfume_df[brand_col].isin(top_brands)]

        # plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid", font_scale=1.3)

        # Create the boxplot
        ax = sns.boxplot(
            x=brand_col,
            y=price_col,
            data=df_filtered,
            #palette="pastel",
            linewidth=2,
            fliersize=8,  # Outlier marker size
            boxprops=dict(alpha=0.7),
        )

        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.title(title, fontsize=18, fontweight="bold")
        plt.xlabel("Brand", fontsize=15, fontweight="medium")
        plt.ylabel("Price ($)", fontsize=15, fontweight="medium")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        # plt.show()

    def show_all_plots(block=True):
        """
        Displays all open matplotlib figures.
        Parameters:
        - block (bool): If True (default), blocks code execution until all plot windows are closed.
            If False, plots are shown but code continues running immediately.
        Need to call this method after all plots with matplotlib are created.
        Example:
            vm.plot_sold_quantity_by_limit(...)
            vm2.plot_sold_quantity_over_time(...)
            show_all_plots()
        This will open all plots simultaneously.
        """
        plt.show(block=block)

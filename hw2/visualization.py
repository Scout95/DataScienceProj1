import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO


class VisualizationModule:
    def __init__(self, figsize=(12, 6), style="seaborn", facecolor="white", title=None):
        self.fig, self.ax = plt.subplots()
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
        plt.figure(figsize=(15, 6))
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
        plt.figure(figsize=(12, 8))
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
        )  # unstack brands to columns for plotting
        plt.figure(figsize=(12, 8))
        # your plotting code here
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, top=0.9)  # increase bottom and top margins

        sold_by_brand_date.plot(marker="o")
        plt.title(title)
        plt.xlabel(xlabelTitle)
        plt.ylabel(yLabelTitle)
        plt.xticks(rotation=45)
        plt.legend(title=legendTitle)

        leg = plt.legend(loc="best")
        leg.set_in_layout(False)  # exclude legend from tight_layout calculations

        plt.tight_layout()
        # plt.show()

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
        perfume_df["lastUpdated"] = pd.to_datetime(
            perfume_df["lastUpdated"], errors="coerce"
        )
        perfume_df = perfume_df.dropna(subset=["lastUpdated", "sold"])
        perfume_df["lastUpdated_naive"] = perfume_df["lastUpdated"].dt.tz_localize(None)

        plt.figure(figsize=(12, 7))
        plt.scatter(perfume_df["lastUpdated_naive"], perfume_df["sold"])
        plt.xlabel("Date")
        plt.ylabel("Sold Quantity")
        plt.xticks(rotation=45)
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

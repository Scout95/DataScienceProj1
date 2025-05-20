from pathlib import Path
from data_loader import DataLoader
from data_processing import DataProcessing
from visualization import VisualizationModule
from missing_values_handler import MissingValuesHandler
import matplotlib.pyplot as plt
from io import StringIO
import os


def main():
    loader = DataLoader()
    base_dir = Path.cwd()  # or Path(__file__).parent

    # # Load data from csv
    print("base_dir:", base_dir)
    current_dir = os.getcwd()
    print("current_dir:", current_dir)
    path_to_file = os.path.join(base_dir, "hw2/dataset/ebay_womens_perfume.csv")

    try:
        df = loader.load_csv(path_to_file)
        print(df.head())
    except RuntimeError as e:
        print(e)
        return

    # Prepare the report about missing values
    data_processing = DataProcessing()
    # check missing values
    missing_data = data_processing.check_missing_values(df)
    print("-------------------> Found missing_data: ------------------>")
    print(missing_data)

    missing_values_handler = MissingValuesHandler(df)
    print(missing_values_handler.missing_values_report())

    # fill missed data by mean value
    filled_by_mean = missing_values_handler.fill_missing_values(df, "mean")
    print("-------------------> Filled missed data by mean value: ------------------>")
    print(filled_by_mean)

    # fill missed data by median value
    filled_by_median = missing_values_handler.fill_missing_values(df, "median")
    print("-------------------> Filled missed data by median value: ----------------->")
    print(filled_by_median)

    # fill missed data by mode (the most common) value
    filled_by_mode = missing_values_handler.fill_missing_values(df, "mode")
    print(
        "------------> Filled missed data by mode (the most common) value: --------->"
    )
    print(filled_by_mode)

    print("-----> The are following columns in Data Frame: ---->")
    print(filled_by_mean.columns)

    # Add histograms
    vm1 = VisualizationModule()
    # vm = VisualizationModule(
    #     figsize=(10, 6), style="seaborn", facecolor="white", title="Price Distribution"
    # )
    subset = filled_by_mean[
        [
            "brand",
            "title",
            "type",
            "price",
            "priceWithCurrency",
            "type",
            "sold",
            "lastUpdated",
        ]
    ].dropna()  # Drop NaN values to avoid errors

    # vm.plot_quantity_by(subset, 'brand', 'sold', 'Total Sold Quantity of Perfume by Brand', 'Brand', 'Sold Quantity')
    vm1.plot_sold_quantity_by_limit(
        subset,
        "brand",
        "sold",
        "Total Sold Quantity of Perfume by Brand",
        "Brand",
        "Sold Quantity",
    )
    vm2 = VisualizationModule()

    # Add linear graphics
    vm2.plot_sold_quantity_over_time(
        subset,
        title="Sold Quantity of Perfume by Brand Over Time",
        xlabelTitle="Date",
        yLabelTitle="Sold Quantity",
        legendTitle="Brand",
    )

    vm3 = VisualizationModule()
    # Add scattering plot
    vm3.plot_scatter_sold_by_brand_over_time(
        subset,
        title="Scatter Plot of Sold Quantity by Brand Over Time",
        xlabelTitle="Date",
        yLabelTitle="Sold Quantity",
        legendTitle="Brand",
    )

    vms = [vm1, vm2, vm3]
    for vm in vms:
        vm.show_all_plots()


if __name__ == "__main__":
    main()

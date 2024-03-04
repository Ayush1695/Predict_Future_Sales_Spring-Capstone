import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Creating lag_features function to create the lags which helps model to predict the future month sales.
def lag_features(df, lags, col_list):
    for col_name in col_list:
        tmp = df[["date_block_num", "shop_id", "item_id", col_name]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = [
                "date_block_num",
                "shop_id",
                "item_id",
                col_name + "_lag_" + str(i),
            ]
            shifted["date_block_num"] += i
            df = pd.merge(
                df, shifted, on=["date_block_num", "shop_id", "item_id"], how="left"
            )
    return df

# Funtion to check the missing value count and percentage
def missing_val_check(data):
    """
    Input::data - A pandas dataframe
    Output::Missing value report by column
    """
    # Missing data check
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat(
        [total, percent * 100], axis=1, keys=["Total", "Percent(%)"]
    )
    return missing_data


def get_boxplots(train_data):
    import matplotlib.pyplot as plt
    # Extract numerical columns
    numeric_columns = list(train_data.select_dtypes(include=['number']).columns)

    # Set up subplots
    num_plots = len(numeric_columns)
    num_rows = (num_plots // 2) + (num_plots % 2)
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 2 * num_rows))

    # Helper function to format column names
    def format_column_name(column_name):
        return column_name.replace("_", " ").capitalize()

    # Plot box plots for each numeric column
    for idx, column in enumerate(numeric_columns):
        row = idx // 2
        col = idx % 2

        try:
            sns.boxplot(train_data[column], ax=axes[row, col], color="red")
            axes[row, col].set_xlabel(format_column_name(column))
        except Exception as e:
            # Handle exceptions, such as when there are not enough subplots for all numeric columns
            print(f"Error plotting {column}: {e}")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()
    
# Define rolling window features
def calculate_rolling_features(x, window_size=3):
    return x.rolling(window=window_size, min_periods=1)

def create_lagged_features(input_df, lag_periods, columns_to_lag):
    for column in columns_to_lag:
        selected_columns = input_df[["date_block_num", "shop_id", "item_id", column]]
        
        for lag in lag_periods:
            shifted_df = selected_columns.copy()
            shifted_df.columns = [
                "date_block_num",
                "shop_id",
                "item_id",
                f"{column}_lag_{lag}",
            ]
            
            shifted_df["date_block_num"] += lag
            
            input_df = pd.merge(
                input_df, shifted_df, on=["date_block_num", "shop_id", "item_id"], how="left"
            )
    
    return input_df

def process_submission_file(X_test,test, model):
    X_test_cp = X_test.copy()
    X_test = X_test.reset_index(drop = True)

    y_test = model.predict(X_test[list(X_train.columns)]).clip(0, 20)
    X_test_cp['item_cnt_month'] = y_test
    
    #X_test_cp['item_cnt_month'] = np.where(X_test_cp['is_new_product']==1,0,X_test_cp['item_cnt_month'])
    
    X_test_cp = X_test_cp.reset_index()
    
    fil_df = train_month_rolled_cp.loc[train_month_rolled_cp["date_block_num"] == 34].reset_index()
    X_test_cp['shop_item_id'] = list(fil_df['shop_item_id'])
    test['shop_item_id'] = test['shop_id'].astype(str) + "_" + test['item_id'].astype(str)
    
    submission = test[['shop_item_id']].merge(X_test_cp[['shop_item_id', 'item_cnt_month']], on = ['shop_item_id'], how = 'left')
    submission = submission.fillna(0)

    submission.drop(['shop_item_id'], axis = 1, inplace = True)
    submission['ID'] = submission.index
    submission = submission[['ID', 'item_cnt_month']]
    return submission
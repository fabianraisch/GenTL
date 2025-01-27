import os
import torch
import numpy as np
import pandas as pd

from src.preprocessing.batch_sampler import *


def read_all_dataframes_from_dir(
        csv_file_dir: str, 
        exclude: list[str], 
        dataframe_limit: int = None, 
        prefix: str = '_'
        ):
        """
        Read all CSV files from a directory and its subdirectories that start with the given prefix into a list of DataFrames.

        Args:
            csv_file_dir (str): The directory containing the CSV files.
            exclude (list, optional): A list of strings to exclude from the csv read in.
            dataframe_limit (int, optional): The maximum number of rows to read from each DataFrame. Defaults to None.
            prefix (str, optional): Only CSV files starting with the prefix are read. Defaults to '_'.
            
        Returns:
            List[pd.DataFrame]: A list of dataframes, each dataframe corresponds to one csv file.
        """

        # If a file is given in the path, only read this one file.
        if os.path.isfile(csv_file_dir):
            dfs = []
            if csv_file_dir.endswith('.csv'):
                df = pd.read_csv(csv_file_dir)[:dataframe_limit]
                dfs.append(df)
            return dfs

        # This covers the case when a directory is given.
        dfs = []
        for (root,_,files) in os.walk(csv_file_dir,topdown=True):
            for file in files:
                if file.startswith(prefix) and file.endswith('.csv'):
                    exclude_current = False
                    for pattern in exclude:
                        if pattern in file:
                            exclude_current = True

                    if exclude_current:
                        continue
                    path_to_csv = os.path.join(root, file)
                    df = pd.read_csv(path_to_csv)[:dataframe_limit]
                    dfs.append(df)

        return dfs      


def preprocess_dataframes(raw_dataframes: list):

    preprocessed_dfs =[]
    for idx, df in enumerate(raw_dataframes):

        # Add a group ID to the dataframe.
        df.insert(1, "group_ids", np.nan, True)
        df["group_ids"] = idx
        df['group_ids'] = df['group_ids'].astype('int')

        # Remove dots as they can not be preprocessed properly in further steps.
        df.columns = df.columns.str.replace('.', '')

        df.reset_index(inplace=True)
        preprocessed_dfs.append(df)
        
    return preprocessed_dfs


def make_train_test_split(
        df: pd.DataFrame, 
        train_ratio: float, 
        val_ratio: float = 0
        ):
    """
    Splits a DataFrame into training, test, and validation sets based on the given ratios.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        train_ratio (float): The ratio of the dataset to include in the training set.
        val_ratio (float): The ratio of the dataset to include in the val set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test dataframes.
    """
    print("Make train test split")
    # Calculate the number of samples for each split
    total_samples = len(df)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    threshold = 1e-8 # necessary for floating point errors
    if abs(1-(train_ratio+val_ratio)) < threshold: # if this is true, then train+test=1
        val_size = int(val_ratio * total_samples) 
        train_size = total_samples - val_size
        
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size] if val_size > 0 else pd.DataFrame()
    test = df.iloc[train_size + val_size:] if test_size > 0 else pd.DataFrame()

    return train, val, test


def make_finetune_split(
    df: pd.DataFrame, 
    finetune_split: list = [0, 30]
):
    """
    Splits a DataFrame into training, test, and validation sets based on the finetune split given.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        finetune_split: List with two elements. First element specifies the start date, second the end date of the train set. The rest will be validation

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test dataframes. Test will be empty here.
    """

    train_start = finetune_split[0]
    train_end = finetune_split[1]

    train = df[(df['day of the year'] >= train_start) & (df['day of the year'] <= train_end)].copy()

    val = df[(df['day of the year'] > train_end)].copy()

    test = pd.DataFrame()

    return train, val, test


def concat_splits(df_splits: list):
    """
    Concatenates training, testing, and validation dataframes while preserving original indexes (for seasonality).
        
    Args:
        dfs (List[pd.DataFrame]): List of tuples containing training, validation, and test dataframes.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Concatenated training, validation, and test dataframes.
    """
    all_train, all_test, all_val = [], [], []

    for df_split in df_splits:
        train, val, test = df_split

        all_train.append(train)

        if not val.empty:
            all_val.append(val)

        if not test.empty:
            all_test.append(test)

    # Concatenate dataframes while keeping original indexes for seasonality
    train_concat = pd.concat(all_train, ignore_index=False)
    val_concat = pd.concat(all_val, ignore_index=False) if all_val else pd.DataFrame()
    test_concat = pd.concat(all_test, ignore_index=False) if all_test else pd.DataFrame()
        
    return train_concat, val_concat, test_concat


def normalize_and_scale(
        train: list, 
        val: list, 
        test: list,
        feature_cols: list,
        target_cols: list,
        scaler):

    from sklearn.preprocessing import MinMaxScaler

    if not scaler:
        scaler_features = MinMaxScaler(feature_range=(0, 1))
        scaler_target = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler_features = scaler[0]
        scaler_target = scaler[1]

    features_without_target = [col for col in feature_cols if col not in target_cols]

    # Fit and transform the feature scaler on the training set
    if not scaler: scaler_features.fit(train[features_without_target])

    train_X_scaled = scaler_features.transform(train[features_without_target])
    train[features_without_target] = train_X_scaled

    if not val.empty:
        val_X_scaled = scaler_features.transform(val[features_without_target])
        val[features_without_target] = val_X_scaled

    if not test.empty:
        test_X_scaled = scaler_features.transform(test[features_without_target])
        test[features_without_target] = test_X_scaled

    if True: # Scale Targets as well
        if not scaler: scaler_target.fit(train[target_cols])

        train_Y_scaled = scaler_target.transform(train[target_cols])
        train[target_cols] = train_Y_scaled

        if not val.empty:
            val_Y_scaled = scaler_target.transform(val[target_cols])
            val[target_cols] = val_Y_scaled

        if not test.empty:
            test_Y_scaled = scaler_target.transform(test[target_cols])
            test[target_cols] = test_Y_scaled

    return train, val, test, scaler_features, scaler_target


def group_by_ids(
        train: list,
        val: list,
        test: list
):
    
    """
    Groups the training, testing, and validation dataframes by their 'group_ids' column.

    Args:
        train (pd.DataFrame): The training dataset to be grouped.
        test (pd.DataFrame): The testing dataset to be grouped. If empty, an empty list is returned.
        val (pd.DataFrame): The validation dataset to be grouped. If empty, an empty list is returned.

    Returns:
        Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]: 
            - grouped_train (list of pd.DataFrame): List of dataframes grouped by 'group_ids' for the training set.
            - grouped_test (list of pd.DataFrame): List of dataframes grouped by 'group_ids' for the testing set. 
            Returns an empty list if the test dataframe is empty.
            - grouped_val (list of pd.DataFrame): List of dataframes grouped by 'group_ids' for the validation set. 
            Returns an empty list if the val dataframe is empty.
    """
    grouped_train = train.groupby('group_ids')
    grouped_val = val.groupby('group_ids') if not val.empty else []
    grouped_test = test.groupby('group_ids') if not test.empty else []

    grouped_train = [grouped_train.get_group(x) for x in grouped_train.groups]
    if grouped_val:
        grouped_val = [grouped_val.get_group(x) for x in grouped_val.groups]
    if grouped_test:
        grouped_test = [grouped_test.get_group(x) for x in grouped_test.groups]

    return grouped_train, grouped_val, grouped_test


def transform_dataset_multivariate(data_X, 
                                   data_Y,
                                   lookback: int = 8,
                                   forecast_horizon: int = 1):
        X, Y = [], []
        for i in range(len(data_X) - lookback - forecast_horizon + 1):
            window = i + lookback
            X.append(data_X[i:window])
            Y.append(data_Y[window:window + forecast_horizon].squeeze(-1))

        # Filter datapoints where the window is off (seasonality)
        return X, Y


def transform_datasets(
        grouped_trains: list,
        grouped_val: list,
        grouped_test: list,
        feature_columns: list,
        target_columns: list,
        lookback: int = 8,
        forecast_horizon: int = 1
):
    
    """
    Processes training, testing, and validation datasets by transforming the data and 
    appending it to corresponding lists. This function also maintains group IDs.

        Args:
            grouped_train (list): List of grouped training data.
            grouped_test (list): List of grouped testing data.
            grouped_val (list): List of grouped validation data.

        Returns:
            tuple: Contains lists of transformed train, test, and validation datasets, and group IDs.
        """
    
    from itertools import zip_longest

    all_train_x, all_train_y = [], []
    all_val_x, all_val_y = [], []
    all_test_x, all_test_y = [], []
    train_groups, val_groups, test_groups = [], [], []
    train_splits, val_splits, test_splits = [], [], []
    for group_id, (train, val, test) in enumerate(zip_longest(grouped_trains, grouped_val, grouped_test, fillvalue=None)):

        # Group again by split_id
        grouped_train_splitid = [train]

        for split_id, train in enumerate(grouped_train_splitid):
            train_x, train_y = transform_dataset_multivariate(
                data_X=torch.tensor(train[feature_columns].astype('float32').values),
                data_Y=torch.tensor(train[target_columns].astype('float32').values),
                lookback=lookback,
                forecast_horizon=forecast_horizon
            )
            all_train_x.extend(train_x)
            all_train_y.extend(train_y)
            train_groups.extend([group_id for _ in train_x])
            train_splits.extend([split_id for _ in train_x])


        if val is not None and not val.empty:
            grouped_val_splitid = [val]

            for split_id, val in enumerate(grouped_val_splitid):

                val_x, val_y = transform_dataset_multivariate(
                    data_X=torch.tensor(val[feature_columns].astype('float32').values),
                    data_Y=torch.tensor(val[target_columns].astype('float32').values),
                    lookback=lookback,
                    forecast_horizon=forecast_horizon
                )
                all_val_x.extend(val_x)
                all_val_y.extend(val_y)
                val_groups.extend([group_id for _ in val_x])
                val_splits.extend([split_id for _ in val_x])
        

        if test is not None and not test.empty:

            grouped_test_splitid = [test]
            
            for split_id, test in enumerate(grouped_test_splitid):
                test_x, test_y = transform_dataset_multivariate(
                    data_X=torch.tensor(test[feature_columns].astype('float32').values),
                    data_Y=torch.tensor(test[target_columns].astype('float32').values),
                    lookback=lookback,
                    forecast_horizon=forecast_horizon
                )
                all_test_x.extend(test_x)
                all_test_y.extend(test_y)
                test_groups.extend([group_id for _ in test_x])
                test_splits.extend([split_id for _ in test_x])
    
    return (all_train_x, all_train_y), (all_val_x, all_val_y), (all_test_x, all_test_y), train_groups, val_groups, test_groups, train_splits, val_splits, test_splits

    

def preprocess_data(
        csv_file_dir: str,
        finetune_split: list,
        feature_columns: list,
        target_columns: list,
        dataframe_limit: int = None,
        train_split: int = 0.7,
        val_split: int = 0.2,
        exclude: list = [],
        lookback: int = 8,
        forecast_horizon: int = 1,
        batch_size: int = 16,
        dataloader_shuffle = False,
        scaler=None

):
    
    raw_dfs = read_all_dataframes_from_dir(csv_file_dir, exclude, dataframe_limit)

    preprocessed_dataframes = preprocess_dataframes(raw_dfs)

    # This is a List[Tuple[pd.DataFrame (TRAIN), pd.DataFrame (VAL), pd.DataFrame (TEST)]]
    if not finetune_split: dataframes_split = [make_train_test_split(df, train_split, val_split) for df in preprocessed_dataframes]
    else: dataframes_split = [make_finetune_split(df, finetune_split) for df in preprocessed_dataframes]

    # Concatinate all the splits into one dataframe for each set.
    train, val, test = concat_splits(dataframes_split)

    # Normalize and scale all sets. Scalers are fit only on the train part.
    train_scaled, val_scaled, test_scaled, scaler_features, scaler_target = normalize_and_scale(train, val, test, feature_columns, target_columns, scaler)

    # Group up the data by their ids (indicating the timeseeries they came from).
    grouped_train, grouped_val, grouped_test = group_by_ids(train_scaled, val_scaled, test_scaled)

    # Transform all the datasets if they exist.
    train_tuple, val_tuple, test_tuple, train_groups, val_groups, test_groups, train_splits, val_splits, test_splits = transform_datasets(
        grouped_train,
        grouped_val,
        grouped_test,
        feature_columns,
        target_columns,
        lookback,
        forecast_horizon
        )
    
    train_dataset = GroupedTimeSeriesDataset(train_tuple[0], train_tuple[1], train_groups, train_splits)
    val_dataset = GroupedTimeSeriesDataset(val_tuple[0], val_tuple[1], val_groups, val_splits)
    test_dataset = GroupedTimeSeriesDataset(test_tuple[0], test_tuple[1], test_groups, test_splits)

    train_batch_sampler = GroupedBatchSampler(train_groups, train_splits, batch_size, dataloader_shuffle)
    val_batch_sampler = GroupedBatchSampler(val_groups, val_splits,batch_size, dataloader_shuffle)
    test_batch_sampler = GroupedBatchSampler(test_groups, test_splits, batch_size, dataloader_shuffle)

    from torch.utils.data import DataLoader
    
    train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    val_data_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler)
    test_data_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)

    return train_data_loader, val_data_loader, test_data_loader, train_batch_sampler, scaler_features, scaler_target
import pandas as pd
from pathlib import Path
import numpy as np
import glob
import os
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


class PlegmaParser:
    """
    Parses the PLEGMA dataset.

    This class handles loading raw data for specified houses, merging daily CSV files,
    resampling the data to a consistent frequency, and saving the processed
    train, validation, and test splits to disk.
    """
    def __init__(self, cfg: DictConfig, output_dir: Path):
        """
        Initializes the PlegmaParser.

        Args:
            cfg (DictConfig): The Hydra configuration object, containing dataset
                              and appliance settings.
            output_dir (Path): The directory where processed data files will be saved.
        """
        self.cfg = cfg
        self.output_dir = output_dir
        self.appliance = cfg.appliance.name
        self.sampling_rate = cfg.dataset.get('sampling', '10s')

        # Construct the absolute path to the raw PLEGMA dataset
        raw_data_root = Path(to_absolute_path(cfg.data.raw_path))
        self.data_location = raw_data_root / cfg.dataset.location

    def _merge_house_files(self, house_id: int) -> pd.DataFrame | None:
        """
        Merges all daily electrical data CSVs for a single house into one DataFrame.

        Args:
            house_id (int): The identifier for the house (e.g., 1 for House_1).

        Returns:
            pd.DataFrame | None: A DataFrame containing the merged data, or None
                                if the house directory or CSV files are not found.
        """
        electric_path = self.data_location / f'House_{house_id}/Electric_data'
        if not electric_path.exists():
            print(f"Warning: Directory not found for House_{house_id}. Skipping.")
            return None
            
        # Find all CSV files, excluding metadata files
        csv_files = sorted(glob.glob(os.path.join(electric_path, "*.csv")))
        valid_files = [f for f in csv_files if "metadata" not in os.path.basename(f)]
        if not valid_files:
            print(f"Warning: No valid data files found for House_{house_id}. Skipping.")
            return None

        # Concatenate all valid files into a single DataFrame
        df = pd.concat((pd.read_csv(f) for f in valid_files), ignore_index=True)
        return df

    def _load_house_data(self, house_id: int) -> pd.DataFrame | None:
        """
        Loads, preprocesses, and resamples data for a single house.

        Args:
            house_id (int): The identifier for the house.

        Returns:
            pd.DataFrame | None: A preprocessed DataFrame with 'aggregate' and
                                appliance columns, or None if the appliance is
                                not present in the house's data.
        """
        df = self._merge_house_files(house_id)
        if df is None or self.appliance not in df.columns:
            print(f"Warning: Appliance '{self.appliance}' not found in House_{house_id}. Skipping.")
            return None

        # Select relevant columns and handle timestamps
        df = df[['timestamp', 'P_agg', self.appliance, 'issues']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Remove data points with known issues
        df = df.drop(index=df[df['issues'] == 1].index, axis=0)
        df = df.rename(columns={'P_agg': 'aggregate'})
        df = df[['aggregate', self.appliance]]
        
        # Resample to a consistent frequency and forward-fill missing values
        df = df.resample(self.sampling_rate).mean().fillna(method='ffill', limit=30)
        return df.dropna().copy()

    def process(self):
        """
        The main processing pipeline for the PLEGMA dataset.

        It loads data for the train, validation, and test splits defined in the
        config, calculates normalization statistics from the training set,
        applies normalization and clipping, and saves the final DataFrames.
        """
        splits = self.cfg.dataset.split[self.appliance]

        # --- Process Training Data ---
        train_dfs = [self._load_house_data(h) for h in splits.train]
        df_train = pd.concat([df for df in train_dfs if df is not None])
        df_train = self._clean_and_clip(df_train)

        # Calculate normalization stats ONLY from the training data
        agg_mean = df_train['aggregate'].mean()
        agg_std = df_train['aggregate'].std()
        print(f"Calculated training stats for '{self.appliance}': Mean={agg_mean:.8f}, Std={agg_std:.8f}")

        # Normalize aggregate and scale appliance data, then save
        df_train['aggregate'] = (df_train['aggregate'] - agg_mean) / agg_std
        df_train[self.appliance] /= self.cfg.dataset.cutoff[self.appliance]
        df_train.to_csv(self.output_dir / self.cfg.data.training_file, index=False)
        print(f"Saved training data to {self.output_dir / self.cfg.data.training_file}")

        # --- Process Validation and Test Data ---
        for split_name, houses in [('validation', splits.val), ('test', splits.test)]:
            split_dfs = [self._load_house_data(h) for h in houses]
            if not split_dfs or all(df is None for df in split_dfs):
                print(f"Warning: No data found for {split_name} split. Skipping.")
                continue
                
            df_split = pd.concat([df for df in split_dfs if df is not None])
            df_split = self._clean_and_clip(df_split)
            
            # Normalize using the training set's mean and std
            df_split['aggregate'] = (df_split['aggregate'] - agg_mean) / agg_std
            df_split[self.appliance] /= self.cfg.dataset.cutoff[self.appliance]
            
            # Save the processed split
            filename = self.cfg.data[f'{split_name}_file']
            df_split.to_csv(self.output_dir / filename, index=False)
            print(f"Saved {split_name} data to {self.output_dir / filename}")


    def _clean_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning and clipping to the power data.

        - Removes negative aggregate power readings.
        - Sets very low power values to 0.
        - Clips both aggregate and appliance power at their predefined maximums.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned and clipped DataFrame.
        """
        df = df[df['aggregate'] > 0]
        df[df < 5] = 0  # Set values below 5W to 0 to reduce noise
        df['aggregate'] = df['aggregate'].clip(upper=self.cfg.dataset.cutoff.aggregate)
        df[self.appliance] = df[self.appliance].clip(upper=self.cfg.dataset.cutoff[self.appliance])
        return df
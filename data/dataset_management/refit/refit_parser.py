# NILM/data/dataset_management/refit/refit_parser.py

import pandas as pd
from pathlib import Path
import numpy as np
from hydra.utils import to_absolute_path

class RefitParser:
    def __init__(self, cfg, output_dir: Path):
        """
        Initializes the REFIT parser with Hydra config.
        Args:
            cfg (DictConfig): The Hydra configuration object.
            output_dir (Path): The directory to save processed files.
        """
        self.cfg = cfg
        self.output_dir = output_dir
        self.appliance = cfg.appliance.name
        self.sampling_rate = cfg.dataset.get('sampling', '8s')

        raw_data_root = Path(to_absolute_path(cfg.data.raw_path))
        self.data_location = raw_data_root / cfg.dataset.location / 'Data'
        self.labels_location = raw_data_root / cfg.dataset.location / 'Labels'

    def _load_house_data(self, house_idx: int) -> pd.DataFrame | None:
        """Loads and preprocesses data for a single house."""
        house_data_loc = self.data_location / f'CLEAN_House{house_idx}.csv'
        label_loc = self.labels_location / f'House{house_idx}.txt'

        if not (house_data_loc.exists() and label_loc.exists()):
            return None

        with open(label_loc) as f:
            house_labels = ['Time', 'Unix'] + f.readline().strip().split(',')

        if self.appliance not in house_labels:
            return None

        appliance_col_index = house_labels.index(self.appliance)
        issues_col_index = house_labels.index('issues')
        
        df = pd.read_csv(house_data_loc, usecols=[0, 1, 2, appliance_col_index, issues_col_index], header=0)
        df.columns = ['Time', 'Unix', 'Aggregate', self.appliance, 'Issues']
        df = df.rename(columns={'Aggregate': 'aggregate'})
        df = df.rename(columns={'Issues': 'issues'})

        df['Unix'] = pd.to_datetime(df['Unix'], unit='s')
        df = df.set_index('Unix')
        df = df.drop(columns=['Time'])
        
        idx_to_drop = df[df['issues'] == 1].index
        df = df.drop(index=idx_to_drop, axis=0)

        df = df.resample(self.sampling_rate).mean().fillna(method='ffill', limit=30)
        return df.dropna().copy()

    def process(self):
        """
        Main processing function for the REFIT dataset.
        """
        splits = self.cfg.dataset.split[self.appliance]
        
        # Process Training Data
        train_dfs = [self._load_house_data(h) for h in splits.train]
        df_train = pd.concat([df for df in train_dfs if df is not None])
        df_train = self._clean_and_clip(df_train)

        agg_mean = df_train['aggregate'].mean()
        agg_std = df_train['aggregate'].std()
        print(f"Calculated training stats for '{self.appliance}': Mean={agg_mean:.8f}, Std={agg_std:.8f}")

        df_train['aggregate'] = (df_train['aggregate'] - agg_mean) / agg_std
        df_train[self.appliance] /= self.cfg.dataset.cutoff[self.appliance]
        df_train[['aggregate', self.appliance]].to_csv(self.output_dir / self.cfg.data.training_file, index=False)

        # Process Validation and Test Data
        for split_name, houses in [('validation', splits.val), ('test', splits.test)]:
            split_dfs = [self._load_house_data(h) for h in houses]
            df_split = pd.concat([df for df in split_dfs if df is not None])
            df_split = self._clean_and_clip(df_split)
            
            df_split['aggregate'] = (df_split['aggregate'] - agg_mean) / agg_std
            df_split[self.appliance] /= self.cfg.dataset.cutoff[self.appliance]

            filename = self.cfg.data[f'{split_name}_file']
            df_split[['aggregate', self.appliance]].to_csv(self.output_dir / filename, index=False)

    def _clean_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies cleaning and clipping to the dataframe."""
        df = df[df['aggregate'] > 0]
        df[df < 5] = 0
        
        df['aggregate'] = df['aggregate'].clip(upper=self.cfg.dataset.cutoff.aggregate)
        df[self.appliance] = df[self.appliance].clip(upper=self.cfg.dataset.cutoff[self.appliance])
        return df
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging

from dataset_management.plegma.plegma_parser import PlegmaParser
from dataset_management.refit.refit_parser import RefitParser

# Set up logging
log = logging.getLogger(__name__)


def process_data(cfg: DictConfig) -> None:
    """
    Main function to orchestrate the data processing pipeline.
    1. Selects the correct parser based on the dataset name.
    2. Instantiates the parser with the Hydra config.
    3. The parser handles loading, cleaning, and saving the data.
    """
    log.info("Starting data processing pipeline...")
    log.info(f"Dataset: {cfg.dataset.name}")
    log.info(f"Appliance: {cfg.appliance.name}")

    # Create the main output directory for processed data
    # Example output: NILM/data/processed/refit/washingmachine/
    output_dir = Path(hydra.utils.to_absolute_path(cfg.data.processed_path)) / cfg.dataset.name / cfg.appliance.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Processed data will be saved to: {output_dir}")

    # Select and instantiate the correct parser based on config
    parser_map = {
        'refit': RefitParser,
        'plegma': PlegmaParser,
        # Add other parsers here as you create them
        # 'ukdale': UkdaleParser,
        # 'redd': ReddParser
    }

    if cfg.dataset.name in parser_map:
        parser = parser_map[cfg.dataset.name](cfg, output_dir)
        parser.process()
    else:
        raise ValueError(f"Unknown dataset specified in config: {cfg.dataset.name}")

    log.info("Data processing complete.")


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.
    """
    print(OmegaConf.to_yaml(cfg))
    process_data(cfg)


if __name__ == "__main__":
    main()
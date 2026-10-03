# Forest CoverType ML Collaboration

Team: Umair and Maaz

This project predicts one of seven forest cover types from geographic and soil measurements. Its goal is to make the dataset, training process, experiments, and released result reproducible by both team members.

## Sources

- Dataset: [UCI Forest CoverType](https://archive.ics.uci.edu/dataset/31/covertype)
- Training-code starting point: [scikit-learn Forest CoverType example](https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_0_24_0.html). The baseline trainer in src/forest_cover/modeling/train.py adapts its data splitting and model training.

## Project layout

- `configs/`: model and experiment settings
- `data/`: raw and processed data, versioned with DVC
- `models/`: trained models, versioned with DVC
- `notebooks/`: exploratory analysis
- `src/forest_cover/`: reusable Python code
- `tests/`: automated checks
- `.github/workflows/`: pull request checks

Setup, reproduction commands, and final metrics will be added as the pipeline is implemented.

## Run the current baseline

uv sync
uv run python -m forest_cover.modeling.train

The current baseline uses a fixed sample and seed. The DVC pipeline and release reproduction steps will be added in later phases.
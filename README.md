# vision-duckweed-tracking

Computer-vision plugin for [science-jubilee](https://github.com/machineagency/science-jubilee):
duckweed frond detection, pose estimation, and RRT-based pickup path
planning. Split out of `science_jubilee/Vision/Duckweed_tracker/` so it can
version and ship independently of the core hardware-control repo.

This repo is a plain pip dependency of `science-jubilee` (not a hardware
`Tool` plugin) — it does not register anything in
`science_jubilee.tools` / the tool registry.

## Layout

```
vision-duckweed-tracking/
├── src/sj_duckweed/
│   ├── ingredients/          # Sacred ingredients: detection, segmentation,
│   │                         # pose estimation, insertion point, RRT planning
│   ├── tracker/              # standalone acquire+track+drive scripts
│   │   ├── duckweed_segment_and_track.py
│   │   └── Jubilee_Duckweed_Tracker.py
│   └── scripts/
│       └── run_duckweed_tracker.py   # Sacred experiment entry point
├── data/
│   └── filtered_images/      # pipeline debug/output images (gitignored except samples)
├── notebooks/
│   └── test_pipeline.ipynb
├── requirements.txt
└── pyproject.toml
```

## Install

```powershell
pip install -e .
pip install -r requirements.txt
```

`science-jubilee` (the `plugin_tools` branch, for `MachineSession`,
`DeckNavigator`, HAL/tool-changer APIs) must be installed separately:

```powershell
pip install -e path/to/science_jubilee
```

## Notebook

`notebooks/test_pipeline.ipynb` exercises the vision pipeline step by step:
image acquisition, float detection, segmentation, duckweed localization,
insertion-point selection, and RRT path planning. Open it in VS Code with the
Jupyter extension, or in another Jupyter-compatible environment, after
installing this project and `science-jubilee`.

## Running

```powershell
python src/sj_duckweed/scripts/run_duckweed_tracker.py
```

Moves the camera to the source well, runs the vision pipeline
(`sj_duckweed.ingredients.pipeline`) to locate a duckweed frond, plans an
RRT pickup path, then drives the inoculator to pick it up and transfer it to
the destination wells.

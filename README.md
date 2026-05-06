## Vision for FSD

Fully vision based path tracking and env mapping for fsd.

This project/repo is a research/testing arena for a full blown FSD workflow comprising of multiple projects ([Monocular SLAM](github.com/gouthamk16/Slam), [Environment Reasoning](github.com/gouthamk16/drive-vlm), [DL Based Actuators](github.com/gouthamk16/xdrive) etc)

## Run an example
```bash
<activate virtual env>
pip3 install -r requirements.txt
python3 main.py -- help
```

### todo
1. Implement extended kalman filters for trajectory mapping and path tracking.
2. Groundwork for 2D to BEV for lane det (bevformer)

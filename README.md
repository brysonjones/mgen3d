# mgen3d

## How to run?
### NeRF Model
1. `cd /path/to/repo/src/libs/mgen3d`
1. `./scripts/build_docker.bash`
1. `./scripts/download_nerf_test_data.bash`
1. `./scripts/run_docker.bash`
1. `cd /ws/mgen3d`
1. `train_full_nerf --data_path=./data/nerf_synthetic/lego/`
1. View `*.gif` outputs in `./outputs/gifs`
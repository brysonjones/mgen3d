
import json
import os
from mgen3d.data.nerf_dataset import NeRFDataset
import torch
from torch.utils.data import DataLoader
import PIL

transforms = {
    "camera_angle_x": 0.6911112070083618,
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    -0.9999021887779236,
                    0.004192245192825794,
                    -0.013345719315111637,
                    -0.05379832163453102
                ],
                [
                    -0.013988681137561798,
                    -0.2996590733528137,
                    0.95394366979599,
                    3.845470428466797
                ],
                [
                    -4.656612873077393e-10,
                    0.9540371894836426,
                    0.29968830943107605,
                    1.2080823183059692
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        {
            "file_path": "./train/r_1",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    -0.9305422306060791,
                    0.11707554012537003,
                    -0.34696459770202637,
                    -1.398659110069275
                ],
                [
                    -0.3661845624446869,
                    -0.29751041531562805,
                    0.8817007541656494,
                    3.5542497634887695
                ],
                [
                    7.450580596923828e-09,
                    0.9475130438804626,
                    0.3197172284126282,
                    1.2888214588165283
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        {
            "file_path": "./train/r_2",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    0.4429636299610138,
                    0.31377720832824707,
                    -0.8398374915122986,
                    -3.385493516921997
                ],
                [
                    -0.8965396881103516,
                    0.1550314873456955,
                    -0.41494810581207275,
                    -1.6727094650268555
                ],
                [
                    0.0,
                    0.936754584312439,
                    0.3499869406223297,
                    1.4108426570892334
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        }
    ]
}

# TODO: generalize the setup and teardown of these tests - there's code duplication here
def test_iteration():
    tmp_dir = "./tmp/"
    root_dir = "./tmp/data/"
    data_split = "train"
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, 'transforms_{}.json'.format(data_split)), 'w') as fp:
        json.dump(transforms, fp)
    os.makedirs(os.path.join(root_dir, data_split), exist_ok=True)
    for i in range(3):
        path = os.path.join(root_dir, data_split, "r_{}.png".format(i))
        PIL.Image.new("RGBA", (5, 5), "WHITE").save(path)
    
    dataset = NeRFDataset(root_dir, data_split)
    assert len(dataset) == 3
    
    for i, sample in enumerate(dataset):
        assert sample['imgs'].shape == (5, 5, 3)
        assert sample['poses'].shape == (4, 4)
        assert sample['H'] == 5
        assert sample['W'] == 5
        assert sample['K'].shape == (3, 3)
        
    # clean up
    os.remove(os.path.join(root_dir, 'transforms_{}.json'.format(data_split)))
    for i in range(3):
        os.remove(os.path.join(root_dir, data_split, "r_{}.png".format(i)))
    os.rmdir(os.path.join(root_dir, data_split))
    os.rmdir(root_dir)
    os.rmdir(tmp_dir)
    
def test_dataloader():
    tmp_dir = "./tmp/"
    root_dir = "./tmp/data/"
    data_split = "train"
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, 'transforms_{}.json'.format(data_split)), 'w') as fp:
        json.dump(transforms, fp)
    os.makedirs(os.path.join(root_dir, data_split), exist_ok=True)
    for i in range(3):
        path = os.path.join(root_dir, data_split, "r_{}.png".format(i))
        PIL.Image.new("RGBA", (5, 5), "WHITE").save(path)
    
    dataset = NeRFDataset(root_dir, data_split)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)
    
    for i, sample in enumerate(dataloader):
        print("sample['H']: ", sample['H'].shape)
        assert sample['imgs'].shape == (3, 5, 5, 3)
        assert sample['poses'].shape == (3, 4, 4)
        assert sample['H'].shape[0] == (3)
        assert torch.eq(sample['H'], torch.tensor([5, 5, 5])).all()
        assert sample['W'].shape[0] == (3)
        assert torch.eq(sample['W'], torch.tensor([5, 5, 5])).all()
        assert sample['K'].shape == (3, 3, 3)
        
    # clean up
    os.remove(os.path.join(root_dir, 'transforms_{}.json'.format(data_split)))
    for i in range(3):
        os.remove(os.path.join(root_dir, data_split, "r_{}.png".format(i)))
    os.rmdir(os.path.join(root_dir, data_split))
    os.rmdir(root_dir)
    os.rmdir(tmp_dir)
    
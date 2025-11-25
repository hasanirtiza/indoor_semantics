### Running the Code
Assuming the sequence is at /my/folder/
You can run it:
```shell 
python main.py --dataset_path /my/folder/rgbd_dataset_freiburg1_room/ 
```

Potentially, you can use oracle poses for splatting and floorplan (upper bounds).
```shell 
python main.py --dataset_path /my/folder/rgbd_dataset_freiburg1_room/ --use_oracle
```
### Installation
```shell 
numpy, opencv, pillow, matplotlib, skimage, scipy  
```
# Quick notes:

Inherently modular problem, keep it modular. Isolate pipeline, easier to optimize params (given time), debug etc  

### Data
* One-to-one association required.
* One can also sample at large intervals.
* Slight motion blur etc. for certain frames.
* Depth has some issues near borders. Not super accurate, also the units.

### PnP
* Classic problem, chaining causes drift.
* Even a constant velocity model can give fairly reasonable results due to the nature of this set.
* Temporal filtering or some sort of BA (local or global). 
* What about something like VGGT ? 


### Splatting
* Drift causes splatting quality to drop a bit.
* Noisy depth.
* Forced isotropy bubbly structure.
* Keeping it lightweight, might want to skip frames and can cause problems.

### Floorplan
* Cascaded problem, errors flow through.
* Tedious params tuning.




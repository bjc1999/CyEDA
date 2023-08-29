# CyEDA: Cycle-object Edge Consistency Domain Adaptation

[ICIP](https://ieeexplore.ieee.org/document/9897493)

### Official pytorch implementation of the paper: "CyEDA: Cycle-object Edge Consistency Domain Adaptation"

#### ICIP 2022

## Results
![result](imgs/val_result.png "result")

## Architecture
![overview](imgs/edge_loss.png "overview")


![gan](imgs/mask_unet.png "gan")

## Environment
1. Clone this repository
   ```
   git clone https://github.com/bjc1999/CyEDA.git
   ```

2. Access the repository folder
   ```
   cd CyEDA
   ```

3. Create virtual environment **python 3.7 recommended**
   ```
   python -m virtualenv env
   ```

4. Activate the environment
   ```
   env/Scripts/activate
   ```

5. Install dependencies
   ```
   pip install -r requirement.txt
   ```

## Training
- Run `train.sh` script
  ```
  bash train.sh
  ```

- Execute `train.py` file within environment
  ```
  python train.py [--parameters]
  ```

## Testing
- Execute `predict.py` file within environment
  ```
  python predict.py [--parameters]
  ```

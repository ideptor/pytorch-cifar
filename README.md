# Train Resnet18

## Prerequisites
- Python 3.10
- PyTorch 2.0.1

## install dependencies
```
pip install -r requirements.txt
```

cpu
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

gpu
pip3 install torch torchvision torchaudio
```

### install pre-commit
```
pip install pre-commit
pre-commit sample-config > .pre-commit-config.yaml
pre-commit install
```

## Prepare dataset
```
python prepare_furniture_image.py -o furniture_images
python prepare_dataset.py -i furniture_images -o dataset-furnitures-3
python main.py -d dataset-furnitures-3/ -e 1000 -b 128 -g

```

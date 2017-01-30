
## Download the model
```
mkdir dataset_dir
cd dataset_dir
wget https://www.doc.ic.ac.uk/~gt108/densereg/model.ckpt-1972331
wget https://www.doc.ic.ac.uk/~gt108/densereg/weight.pkl
wget https://www.doc.ic.ac.uk/~gt108/densereg/train.tfrecords
```

## Train the model
```python train.py --train_dir=ckpt/pose --batch_size=4 --initial_learning_rate=0.0001 --dataset_dir=dataset_dir```

## Run the demo

[Demo](https://github.com/trigeorgis/densereg/blob/master/Demo%20Pose%20Machine.ipynb)

RANK = 5

mpirun -np $RANK python3 ../experiment.py --dataset FashionMNIST --compress topk30 --optimizer DADAM --comm-set x g --lr 0.001 --batch-size 8 --variety index --model LeNet5 --epochs 100 --device gpu --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset FashionMNIST --compress topk30 --optimizer DAMSCo --comm-set x --lr 0.001 --batch-size 8 --variety index --model LeNet5 --epochs 100 --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset FashionMNIST --compress topk30 --optimizer DAdaGrad --comm-set x g --lr 0.01 --batch-size 8 --variety index --model LeNet5 --epochs 100 --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset FashionMNIST --compress topk30 --optimizer DaSHCo --comm-set x g --lr 0.02 --batch-size 8 --variety index --model LeNet5 --epochs 100 --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset FashionMNIST --compress topk30 --optimizer CDProxSGD --comm-set x g --lr 0.02 --batch-size 8 --variety index --model LeNet5 --epochs 100 --rank $RANK
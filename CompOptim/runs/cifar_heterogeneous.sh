RANK = 5

mpirun -np $RANK python3 ../experiment.py --dataset CIFAR10 --compress topk40 --optimizer DADAM --comm-set x g --lr 0.001 --batch-size 32 --variety label --model fixup_resnet20 --epochs 300 --device gpu --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset CIFAR10 --compress topk40 --optimizer DAMSCo --comm-set x --lr 0.001 --batch-size 32 --variety label --model fixup_resnet20 --epochs 300 --device gpu --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset CIFAR10 --compress topk40 --optimizer DAdaGrad --comm-set x g --lr 0.01 --batch-size 32 --variety label --model fixup_resnet20 --epochs 300 --device gpu --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset CIFAR10 --compress topk40 --optimizer DaSHCo --comm-set x g --lr 0.02 --batch-size 32 --variety label --model fixup_resnet20 --epochs 300 --device gpu --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset CIFAR10 --compress topk40 --optimizer CDProxSGD --comm-set x g --lr 0.02 --batch-size 32 --variety label --model fixup_resnet20 --epochs 300 --device gpu --rank $RANK
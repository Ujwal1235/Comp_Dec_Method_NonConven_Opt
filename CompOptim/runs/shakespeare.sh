RANK = 4

mpirun -np $RANK python3 ../experiment.py --dataset Shakespeare --compress topk55 --optimizer DADAM --comm-set x g --lr 0.0001 --batch-size 128 --variety index --model nanoGPT --epochs 650 --device gpu --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset Shakespeare --compress topk55 --optimizer DAMSCo --comm-set x --lr 0.0001 --batch-size 128 --variety index --model nanoGPT --epochs 650 --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset Shakespeare --compress topk55 --optimizer DAdaGrad --comm-set x g --lr 0.0001 --batch-size 128 --variety index --model nanoGPT --epochs 650 --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset Shakespeare --compress topk55 --optimizer DaSHCo --comm-set x g --lr 0.02 --batch-size 128 --variety index --model nanoGPT --epochs 650 --rank $RANK

mpirun -np $RANK python3 ../experiment.py --dataset Shakespeare --compress topk55 --optimizer CDProxSGD --comm-set x g --lr 0.02 --batch-size 128 --variety index --model nanoGPT --epochs 650 --rank $RANK
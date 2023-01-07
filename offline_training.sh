python run.py --algo=density-ucr-offline --task=walker2d --ds=random --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=halfcheetah --ds=random --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=hopper --ds=random --gpu-no=0  --seed=10 --save=True --train=True

python run.py --algo=density-ucr-offline --task=walker2d --ds=medium --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=halfcheetah --ds=medium --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=hopper --ds=medium --gpu-no=0  --seed=10 --save=True --train=True

python run.py --algo=density-ucr-offline --task=walker2d --ds=medium-replay --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=halfcheetah --ds=medium-replay --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=hopper --ds=medium-replay --gpu-no=0  --seed=10 --save=True --train=True


python run.py --algo=density-ucr-offline --task=walker2d --ds=medium-expert --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=halfcheetah --ds=medium-expert --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=hopper --ds=medium-expert --gpu-no=0  --seed=10 --save=True --train=True


python run.py --algo=density-ucr-offline --task=walker2d --ds=expert --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=halfcheetah --ds=expert --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=hopper --ds=expert --gpu-no=0  --seed=10 --save=True --train=True

python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=umaze-dense --gpu-no=0 --seed=10 --save=True --train=True
python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=medium-dense --gpu-no=0  --seed=10 --save=True --train=True
python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=large-dense --gpu-no=0  --seed=10 --save=True --train=True

python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=umaze --gpu-no=0 --seed=10 --save=True --train=True
python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=medium --gpu-no=0  --seed=10 --save=True --train=True
python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=large --gpu-no=0  --seed=10 --save=True --train=True

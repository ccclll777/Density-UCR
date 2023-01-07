python run.py --algo=density-ucr-online --task=walker2d --ds=random --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=halfcheetah --ds=random --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=hopper --ds=random --gpu-no=0  --seed=10 --save=True --train=True

python run.py --algo=density-ucr-online --task=walker2d --ds=medium --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=halfcheetah --ds=medium --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=hopper --ds=medium --gpu-no=0  --seed=10 --save=True --train=True


python run.py --algo=density-ucr-online --task=walker2d --ds=medium-replay --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=halfcheetah --ds=medium-replay --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=hopper --ds=medium-replay --gpu-no=0  --seed=10 --save=True --train=True

python run.py --algo=density-ucr-online --task=walker2d --ds=medium-expert --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=halfcheetah --ds=medium-expert --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=hopper --ds=medium-expert --gpu-no=0  --seed=10 --save=True --train=True

python run.py --algo=density-ucr-online --task=walker2d --ds=expert --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=halfcheetah --ds=expert --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=hopper --ds=expert --gpu-no=0  --seed=10 --save=True --train=True
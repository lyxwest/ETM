if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

python -m torch.distributed.launch --master_port 1238 --nproc_per_node=8 \
    test.py --config ${config} --weights ${weight} ${@:3}
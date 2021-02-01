# $1 env num - 1
cp -r battle-framework battle-framework-copy

cd battle-framework

#nohup /home/inksci/miniconda3/envs/py3-tfv5-keras2/bin/python train_deepfire.py >train_deepfire-output 2>&1 & echo $! >> ../pid.txt
nohup /home/inksci/miniconda3/envs/py3-tfv5-keras2/bin/python train_deepfire.py 1>/dev/null 2>train_deepfire-output & echo $! >> ../pid.txt

while [ ! -f "agent/deepfire/model/checkpoint" ]
do
sleep 1s
echo "."
ls
done

cat "agent/deepfire/model/checkpoint"



nohup /home/inksci/miniconda3/envs/py3-tfv5-keras2/bin/python run_deepfire.py --env=0 1>/dev/null 2>run_deepfire-output & echo $! >> ../pid.txt

echo 1002

for i in $(seq 1 $1)
do
cd ..
cp -r battle-framework-copy "battle-framework-env"$i
cd "battle-framework-env"$i

nohup /home/inksci/miniconda3/envs/py3-tfv5-keras2/bin/python run_deepfire.py --env=$i 1>/dev/null 2>run_deepfire-output & echo $! >> ../pid.txt

done

cd ..


echo "pid:"
cat pid.txt

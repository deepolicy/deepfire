# stop

cd battle-framework/agent/deepfire

mv data data-stop

echo 1001


for i in $(seq 1 $1)
do
cd ../../../

ls
cd "battle-framework-env"$i"/agent/deepfire"
mv data data-stop
echo 1002
echo $i
done

cd ../../../

echo "done!"
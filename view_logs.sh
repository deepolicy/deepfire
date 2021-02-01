# view logs.txt

cd battle-framework

echo 0
cat logs.txt
echo '------------------------------'


for i in $(seq 1 $1)
do
    cd ../

    cd "battle-framework-env"$i

    echo $i
    cat logs.txt
    echo '------------------------------'
done

cd ../

echo "done!"
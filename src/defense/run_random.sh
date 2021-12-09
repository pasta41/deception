for j in 1 2 3 4 5 6 7 8 9 10
do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        python main.py nes $(($j*10+$i))
    done
done

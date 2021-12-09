for i in 0.5 0.25 0.1 5e-2 2.5e-2 1e-2 5e-3 2.5e-3 1e-3 5e-4 2.5e-4 1e-4
do
    for j in 1 2 3 4 5
    do
        python main.py $i nes $j
    done
done

for i in 0.5 0.25 0.1 5e-2 2.5e-2 1e-2 5e-3 2.5e-3 1e-3 5e-4 2.5e-4 1e-4
do
    for j in 1 2 3 4 5
    do
        python main.py $i HB $j
    done
done


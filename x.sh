for layer1_out in 30 50 100
do

for delta in 5 10 20 30
do

for sim in nnlsw
do

for alpha in 0.1
do


echo layer1_out $layer1_out, delta $delta, sim $sim, alpha $alpha
python eval.py --layer1_out $layer1_out --delta $delta --alpha $alpha --sim $sim 

done

done

done

done

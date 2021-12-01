export DATA=../cnn-dailymail/finished_files
export METEOR=/home/victoria/summary/fast-abs-rl/meteor-1.5/meteor-1.5.jar
export ROUGE=/home/victoria/summary/fast-abs-rl/ROUGE-1.5.5/

mkdir -p tmp/output
#echo "making references ..."
#python3 make_eval_references.py

for label in test val
do
	echo "evaluating $label data ..."
	for beam in 1 5
	do
		echo "decoding $label data (beam = $beam) ..."
		python3 decode_full_model.py --path=tmp/output/${label}/beam-${beam} --model_dir=tmp/rl/ --beam=$beam --$label
		for metric in rouge meteor
		do
			echo "beam = $beam, metric = $metric"
			python3 eval_full_model.py --$metric --decode_dir=tmp/output/${label}/beam-${beam} > logs/${label}_beam-$beam.$metric
		done
	done
done

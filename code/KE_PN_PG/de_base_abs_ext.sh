export DATA=../cnn-dailymail/finished_files
export METEOR=/home/victoria/summary/fast-abs-rl/meteor-1.5/meteor-1.5.jar
export ROUGE=/home/victoria/summary/fast-abs-rl/ROUGE-1.5.5/

mkdir -p tmp/baseline/ExtAbs_1_1_5
#echo "making references ..."
#python3 make_eval_references.py

for label in test
do
	echo "evaluating $label data ..."
	for beam in 1 
	do
		echo "decoding $label data (beam = $beam) ..."
        rm -rf tmp/baseline/ExtAbs_1_1_5
		python3 decode_abs_ext.py --path=tmp/baseline/ExtAbs_1_1_5/${label}/beam-${beam} --abs_dir=tmp/abstractor_85_30/ --ext_dir=tmp/extractor_loss_1_1 --beam=$beam --$label
	done
done


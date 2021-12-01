export DATA=../cnn-dailymail/finished_files
export METEOR=/home/victoria/summary/fast-abs-rl/meteor-1.5/meteor-1.5.jar
export ROUGE=/home/victoria/summary/fast-abs-rl/ROUGE-1.5.5/

rm -rf tmp/baseline/abs_85_30/test/beam-1
mkdir -p tmp/baseline/abs_85_30
#echo "making references ..."
#python3 make_eval_references.py

for label in test
do
	echo "evaluating $label data ..."
	echo "decoding $label data (beam = 1) ..."
	CUDA_VISIBLE_DEVICES=0 python3 decode_abs.py --path=tmp/baseline/abs_85_30/${label}/beam-1 --abs_dir=tmp/abstractor_85_30/ --$label
done


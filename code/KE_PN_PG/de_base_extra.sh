export DATA=../cnn-dailymail/finished_files
export METEOR=/home/victoria/summary/fast-abs-rl/meteor-1.5/meteor-1.5.jar
export ROUGE=/home/victoria/summary/fast-abs-rl/ROUGE-1.5.5/

mkdir -p tmp/baseline/ext_loss_1_0.5
#echo "making references ..."
#python3 make_eval_references.py

for label in test
do
	echo "evaluating $label data ..."
	echo "decoding $label data (beam = 1) ..."
    rm -rf tmp/baseline/ext_loss_1_0.5
	CUDA_VISIBEL_DEVICES=0 python3 decode_ext.py --path=tmp/baseline/ext_loss_1_0.5/${label}/beam-1 --ext_dir=tmp/extractor_loss_1_0.5/ --$label
done


export DATA=../cnn-dailymail/finished_files/
# pretrain word2vec
#mkdir -p tmp
#python3 -u train_w2v_woE.py --path=tmp/word2vec/
# make the pseudo-labels
#python3 make_extraction_test.py
# train abstractor and extractor
rm -rf tmp/abstractor_85_30/
CUDA_VISIBLE_DEVICES=0 python3 -u train_abstractor_woE.py --path=tmp/abstractor_85_30/ --w2v=tmp/word2vec/word2vec.128d.226k.bin
#python3 train_extractor_ml.py --path=tmp/extractor/ --w2v=tmp/word2vec/word2vec.128d.222k.bin
# train full RL
#python3 train_full_rl.py --path=tmp/rl/ --abs_dir=tmp/abstractor/ --ext_dir=tmp/extractor/
# decode
#mkdir -p tmp/output
#python3 decode_full_model.py --path=tmp/output/decoded --model_dir=tmp/rl/ --beam=1 --test
# make reference files for evaluation
#python3 make_eval_references.py
# run evaluation
#export METEOR=/home/victoria/summary/fast-abs-rl/meteor-1.5/meteor-1.5.jar
#export ROUGE=/home/victoria/summary/fast-abs-rl/ROUGE-1.5.5/
#pyhon3 eval_full_model.py --rouge --decode_dir=tmp/output/decoded

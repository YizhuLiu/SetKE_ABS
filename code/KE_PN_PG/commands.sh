export DATA=../cnn-dailymail/finished_files/
# pretrain word2vec
#mkdir -p tmp
#python3 train_w2v_woE.py --path=tmp/word2vec/
# make the pseudo-labels
#python3 make_extraction_test.py
# train abstractor and extractor
#python3 train_abstractor_woE.py --path=tmp/abstractor/ --w2v=tmp/word2vec/word2vec.128d.222k.bin
#rm -rf tmp/extractor
#CUDA_VISIBLE_DEVICES=1 python3 train_extractor_ml.py --path=tmp/extractor_loss_1_0.5/ --w2v=tmp/word2vec/word2vec.128d.222k.bin
# train full RL
#rm -rf tmp/rl
#mkdir -p tmp/rl_15_5
#rm -rf tmp/rl_15_5
#CUDA_VISIBLE_DEVICES=1 python3 -u train_full_rl.py --path=tmp/rl_15_5/ --abs_dir=tmp/abstractor_85_30/ --ext_dir=tmp/extractor_loss_1_0.5/ 
# decode
mkdir -p tmp/output
rm -rf tmp/output/ExtAbsRL_loss_reward_15_5_10_3
python3 decode_full_model.py --path=tmp/output/ExtAbsRL_loss_reward_15_5_10_3 --model_dir=tmp/rl_15_5/ --beam=1 --test
# make reference files for evaluation
#python3 make_eval_references.py
# run evaluation
#export METEOR=/home/victoria/summary/fast-abs-rl/meteor-1.5/meteor-1.5.jar
#export ROUGE=/home/victoria/summary/fast-abs-rl/ROUGE-1.5.5/
#pyhon3 eval_full_model.py --rouge --decode_dir=tmp/output/decoded

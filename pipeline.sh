export GLUE_DIR=glue_data
export TASK_DIR=MRPC
export MODEL=bert-base-uncased

python run_classifier.py \
  --task_name $TASK_DIR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_DIR\
  --bert_model $MODEL \
  --max_seq_length 128 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir outputs/"$TASK_DIR"

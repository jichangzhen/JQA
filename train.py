import datetime, os, sys, nltk, logging
import tensorflow as tf
import pickle as pkl
from pprint import pprint
from collections import deque
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_helper import Vocabulary, WordTable, NodeVocab, NodeTable
from hierarchical_s2s import DiscourseAwareHierarchicalSeq2seqBase
from model import SelfattlSeq2seqBase
from optimization import create_optimizer
from nltk.translate.bleu_score import SmoothingFunction
import rouge.rouge_score as rouge_score

sys.path.append("..")

# Data loading params
tf.flags.DEFINE_string("train_data_file", "../../shared/selfdata/jbmdata/train_with_path", "Data source for training.")
tf.flags.DEFINE_string("dev_data_file", "../../shared/selfdata/jbmdata/val_with_path", "Data source for validation.")
tf.flags.DEFINE_string("test_data_file", "../../shared/selfdata/jbmdata/test_with_path", "Data source for testing.")
tf.flags.DEFINE_string("word_embedding_file", "../../shared/selfdata/vocab/voc_w2v", "Pre train embedding file.")
tf.flags.DEFINE_string("vocab_model_file", "../../shared/selfdata/vocab/vocab.model", "Vocab model file.")
tf.flags.DEFINE_string("train_pkl_file", "../../shared/selfdata/jbmdata/jbm_data.pkl", "Data source for article classes.")
tf.flags.DEFINE_boolean("fine_tuning", False, "fine_tuning from pretrained lm files")
tf.flags.DEFINE_boolean("continue_training", True, "continue training from restore, or start from scratch")
tf.flags.DEFINE_string("pre_train_lm_checkpoint_path", " ", "Checkpoint file path from pre trained language model.'")
tf.flags.DEFINE_string("checkpoint_path", "./check/2020-04-24_19-16-01/checkpoints/model-62500",
                       "Checkpoint file path without extension, as list in file 'checkpoints'")
tf.flags.DEFINE_string("pre_trained_word_embeddings", "", "pre_trained_word_embeddings")
tf.flags.DEFINE_string("model", "selfatt", "seq2seq/selfatt")

# Data sample bound
tf.flags.DEFINE_integer("lower_bound", 3000, "lower bound frequency for over-sampling (default: 3,000)")
tf.flags.DEFINE_integer("upper_bound", 1000000, "upper bound frequency for sub-sampling (default: 100,000)")
tf.flags.DEFINE_integer("over_sample_times", 0, "over_sample_times (default: 1)")

# Embedding params
tf.flags.DEFINE_integer("edim", 300, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("embedding_dense_size", 300, "Dimensionality of word embedding dense layer (default: 300)")
tf.flags.DEFINE_boolean("use_role_embedding", True, "Use role embedding or not  (default:True)")
tf.flags.DEFINE_integer("role_num", 5, "How many roles  (default: 3)")
tf.flags.DEFINE_integer("role_edim", 100, "Dimensionality of role embedding  (default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_boolean("pointer_gen", True, " ")
tf.flags.DEFINE_boolean("use_transformer_linear_projection", True, " ")
tf.flags.DEFINE_integer("use_version", 2, " ")
tf.flags.DEFINE_integer("max_sequence_length", 40, "Max sentence sequence length (default: 40)")
tf.flags.DEFINE_integer("num_classes", 41, "Number of classes (default: 41)")
tf.flags.DEFINE_integer("vocab_size", 20001, "Words in total in Vocab  (default: 160000)")
tf.flags.DEFINE_integer("pointer_vocab_size", 20001, "Words in total in Pointer Vocab  (default: 160000)")

# transformer used
tf.flags.DEFINE_integer("transformer_layers", 2, "Transformer layers (default: 4)")
tf.flags.DEFINE_integer("sen_transformer_layers", 0, "Sentence Level Transformer layers (default: 1)")
tf.flags.DEFINE_integer("heads", 8, "multi-head attention (default: 4)")
tf.flags.DEFINE_integer("intermediate_size", 1000, "Intermediate size (default: 1000)")

# rnn used
tf.flags.DEFINE_integer("rnn_hidden_size", 200, "rnn hidden size (default: 300)")
tf.flags.DEFINE_integer("rnn_layer_num", 1, "rnn layer num (default: 1)")
tf.flags.DEFINE_integer("rnn_attention_size", 400, "rnn attention dense layer size (default: 300)")
tf.flags.DEFINE_integer("rnn_output_mlp_size", 500, "rnn output mlp size (default: 500)")
tf.flags.DEFINE_integer("num_k", 15, "drnn window size (default: 15)")
tf.flags.DEFINE_integer("ram_gru_size", 300, "recurrent attention gru cell size (default: 300)")
tf.flags.DEFINE_integer("ram_times", 4, "recurrent attention times (default: 4)")
tf.flags.DEFINE_integer("ram_hidden_size", 300,
                        "recurrent attention final episode attention hidden size (default: 300)")

# cnn used
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '1,2,3,4,5')")
tf.flags.DEFINE_integer("fc1_dense_size", 512, "fc size before output layer (default: 2048)")
tf.flags.DEFINE_string("num_filters", "64,64,64,64", "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("init_std", 0.01, "Init std value for variables (default: 0.01)")
tf.flags.DEFINE_float("input_noise_std", 0.05, "Input for noise  (default: 0.01)")

tf.flags.DEFINE_float("max_grad_norm", 10, "clip gradients to this norm (default: 10)")
tf.flags.DEFINE_string("activation_function", "relu",
                       "activation function used (default: relu) ")  # relu swish elu crelu tanh gelu

# Training parameters
tf.flags.DEFINE_integer("max_decoder_steps", 100, "max_decoder_steps")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 5e-4, "initial learning rate (default: 1e-3)")
tf.flags.DEFINE_float("decay_rate", 0.9, "learning rate decay rate (default: 0.7)")
tf.flags.DEFINE_integer("decay_step", 500, "learning rate decay step (default: 20000)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on valid set after this many steps (default: 1000)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("warm_up_steps_percent", 0.05, "Warm up steps percent (default: 5%)")

# Knowledge parameters
tf.flags.DEFINE_boolean("use_knowledge", True, "Use knowledge embedding or not True or False (default:True)")
tf.flags.DEFINE_integer("max_path_length", 5, "Max path sequence length (default: 5)")
tf.flags.DEFINE_string("pre_node_embeddings", " ", "pre_node_embeddings")
tf.flags.DEFINE_string("node_embedding_file", "../../shared/selfdata/path/node_emb", "Pre train node embedding file.")
tf.flags.DEFINE_string("node_vocab_file", "../../shared/selfdata/path/node", "kb node path")

tf.flags.DEFINE_string("cuda_device", "0,1", "GPU used")

FLAGS = tf.flags.FLAGS
FLAGS.filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
FLAGS.num_filters = list(map(int, FLAGS.num_filters.split(",")))

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
from data_helper import CustomSegmentor

segmentor = CustomSegmentor()


def train(data):
	train_data_set = data["train_data_set"]
	train_data_set_handout = data["train_data_set_handout"]
	valid_data_set = data["valid_data_set"]
	test_data_set = data["test_data_set"]
	
	train_num_samples = len(train_data_set)
	batch_num = (train_num_samples * FLAGS.num_epochs) // FLAGS.batch_size + 1
	
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	session_conf.gpu_options.allow_growth = True
	
	session_conf = tf.ConfigProto()
	session_conf.gpu_options.allow_growth = True
	
	with tf.Session(graph=tf.Graph(), config=session_conf) as session:
		if FLAGS.model == "seq2seq":
			model = DiscourseAwareHierarchicalSeq2seqBase(FLAGS)
		elif FLAGS.model == "selfatt":
			model = SelfattlSeq2seqBase(FLAGS)
		else:
			raise Exception("unexpected model")
		
		# bert loss
		train_op, learning_rate, global_step = create_optimizer(model.loss, FLAGS.learning_rate,
		                                                        num_train_steps=batch_num, num_warmup_steps=int(
				batch_num * FLAGS.warm_up_steps_percent), use_tpu=False)
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch normalization
		# Output directory for models
		timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "check", timestamp))
		print("Writing to {}\n".format(out_dir))
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		logging.basicConfig(level=logging.DEBUG,
		                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
		                    datefmt='%a, %d %b %Y %H:%M:%S',
		                    filename=os.path.join(checkpoint_dir, "log.txt"),
		                    filemode='w+')
		logging.info(FLAGS)
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=FLAGS.num_checkpoints)
		
		# Initialize all variables
		if FLAGS.continue_training:
			initialize_op = tf.variables_initializer(
				[x for x in tf.global_variables() if x not in tf.trainable_variables()])
			session.run(initialize_op)
			saver.restore(session, FLAGS.checkpoint_path)
		else:
			session.run(tf.global_variables_initializer())
		
		def _tokenid_to_natural_sentence_ids(tokenids):
			tokenids = [x for x in tokenids if x not in (0, 1)]
			min_eos_index = tokenids.index(2) if 2 in tokenids else -1
			
			if min_eos_index > 0:
				tokenids = tokenids[:min_eos_index]
			
			return tokenids
		
		AVAILABLE_METRICS = {
			"rouge-1": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 1),
			"rouge-2": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 2),
			"rouge-3": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 3),
			"rouge-l": lambda hyp, ref:
			rouge_score.rouge_l_summary_level(hyp, ref),
		}
		
		def cal_rouge(hyp, ref):
			scores = {}
			for k, fn in AVAILABLE_METRICS.items():
				scores[k] = fn(hyp, ref)
			return scores
		
		def _do_train_step(input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch,
		                   path_batch, path_lens_batch, decoder_input_x_batch, decoder_output_x_batch,
		                   decoder_lens_batch):
			"""
			A single training step
			"""
			if FLAGS.use_version != 1:
				decoder_input_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_x_batch,
				                                                                      maxlen=FLAGS.max_decoder_steps,
				                                                                      padding="post",
				                                                                      truncating="post",
				                                                                      value=0)
				decoder_output_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_x_batch,
				                                                                       maxlen=FLAGS.max_decoder_steps,
				                                                                       padding="post",
				                                                                       truncating="post",
				                                                                       value=0)
			feed_dict = {model.input_x: input_x_batch,
			             model.input_role: input_role_batch,
			             model.input_sample_lens: input_sample_lens_batch,
			             model.input_sentences_lens: input_sentences_lens_batch,
			             model.path: path_batch,
			             model.path_lens: path_lens_batch,
			             model.decoder_inputs: decoder_input_x_batch,
			             model.decoder_outputs: decoder_output_x_batch,
			             model.decoder_lengths: decoder_lens_batch,
			             model.dropout_keep_prob: FLAGS.dropout_keep_prob,
			             model.training: True
			             }
			fetches = [update_ops, train_op, learning_rate, global_step, model.loss, model.decoder_loss,
			           model.infer_predicts]
			_, _, lr, step, loss, decoder_loss, batch_seq2seq_predict = session.run(fetches=fetches,
			                                                                        feed_dict=feed_dict)
			
			bleu_scores = []
			rouge1 = []
			rouge2 = []
			rouge3 = []
			rougel = []
			for p, g in zip(batch_seq2seq_predict.tolist(), decoder_output_x_batch.tolist()):
				gs = _tokenid_to_natural_sentence_ids(g)
				ps = _tokenid_to_natural_sentence_ids(p)
				gs, ps = list(map(lambda x: str(x), gs)), list(map(lambda x: str(x), ps))
				score = nltk.bleu([gs], ps, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4),
				                  smoothing_function=SmoothingFunction().method1)
				bleu_scores.append(score)
				r_scores = cal_rouge([" ".join(ps)], [" ".join(gs)])
				rouge1.append(r_scores["rouge-1"]["f"])
				rouge2.append(r_scores["rouge-2"]["f"])
				rouge3.append(r_scores["rouge-3"]["f"])
				rougel.append(r_scores["rouge-l"]["f"])
			
			time_str = datetime.datetime.now().isoformat()
			print(
				"\n{}:step {}, lr {:g}, loss {:g}, decoder_loss {:g},  bleu_score {:g}, rouge1 {:g}, rouge2 {:g}, rouge3 {:g}, rougel {:g} "
					.format(time_str, step, lr, np.mean(loss), np.mean(decoder_loss), np.mean(bleu_scores) * 100,
				            np.mean(rouge1) * 100, np.mean(rouge2) * 100, np.mean(rouge3) * 100, np.mean(rougel) * 100
				            )
			)
		
		def _do_valid_step(data_set):
			"""
			Evaluates model on a valid set
			"""
			num_samples = len(data_set)
			div = num_samples % FLAGS.batch_size
			batch_num = num_samples // FLAGS.batch_size + 1 if div != 0 else num_samples // FLAGS.batch_size
			
			tf_data_set = tf.data.Dataset.from_generator(lambda: data_set, (
				tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
			                                             ).padded_batch(FLAGS.batch_size,
			                                                            padded_shapes=(tf.TensorShape([None, None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([None, None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([])),
			                                                            padding_values=(
				                                                            0, 0, 0, 0, 0, 0, 0, 0, 0))  # pad index 0
			
			valid_iterator = tf_data_set.make_one_shot_iterator()
			valid_one_batch = valid_iterator.get_next()
			
			losses = []
			decoder_losses = []
			seq2seq_predicts = []
			seq2seq_y_true = []
			for _ in range(batch_num):
				input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch, path_batch, path_lens_batch, decoder_input_x_batch, decoder_output_x_batch, decoder_lens_batch = session.run(
					valid_one_batch)
				
				if FLAGS.use_version != 1:
					decoder_input_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_x_batch,
					                                                                      maxlen=FLAGS.max_decoder_steps,
					                                                                      padding="post",
					                                                                      truncating="post",
					                                                                      value=0)
					decoder_output_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_x_batch,
					                                                                       maxlen=FLAGS.max_decoder_steps,
					                                                                       padding="post",
					                                                                       truncating="post",
					                                                                       value=0)
				feed_dict = {model.input_x: input_x_batch,
				             model.input_role: input_role_batch,
				             model.input_sample_lens: input_sample_lens_batch,
				             model.input_sentences_lens: input_sentences_lens_batch,
				
				             model.path: path_batch,
				             model.path_lens: path_lens_batch,
				
				             model.decoder_inputs: decoder_input_x_batch,
				             model.decoder_outputs: decoder_output_x_batch,
				             model.decoder_lengths: decoder_lens_batch,
				             model.dropout_keep_prob: 1.0,
				             model.training: False
				             }
				fetches = [model.loss, model.decoder_loss, model.infer_predicts]
				loss, decoder_loss, batch_seq2seq_predict = session.run(
					fetches=fetches,
					feed_dict=feed_dict)
				losses.append(loss)
				decoder_losses.append(decoder_loss)
				seq2seq_predicts.extend(batch_seq2seq_predict.tolist())
				seq2seq_y_true.extend(decoder_output_x_batch.tolist())
			
			rouge1 = []
			rouge2 = []
			rouge3 = []
			rougel = []
			bleu_scores = []
			for p, g in zip(seq2seq_predicts, seq2seq_y_true):
				gs = _tokenid_to_natural_sentence_ids(g)
				ps = _tokenid_to_natural_sentence_ids(p)
				gs, ps = list(map(lambda x: str(x), gs)), list(map(lambda x: str(x), ps))
				score = nltk.bleu([gs], ps, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4),
				                  smoothing_function=SmoothingFunction().method1)
				bleu_scores.append(score)
				r_scores = cal_rouge([" ".join(ps)], [" ".join(gs)])
				rouge1.append(r_scores["rouge-1"]["f"])
				rouge2.append(r_scores["rouge-2"]["f"])
				rouge3.append(r_scores["rouge-3"]["f"])
				rougel.append(r_scores["rouge-l"]["f"])
			
			mean_loss = np.mean(losses)
			mean_decoder_loss = np.mean(decoder_losses)
			bleu_score = np.mean(bleu_scores) * 100
			logging.info(
				"num_samples {}, loss {:g}, decoder_loss {:g}, bleu_score {:g}, rouge1 {:g}, rouge2 {:g}, rouge3 {:g}, rougel {:g} ".format(
					num_samples, mean_loss, mean_decoder_loss, bleu_score,
					np.mean(rouge1) * 100, np.mean(rouge2) * 100, np.mean(rouge3) * 100, np.mean(rougel) * 100
				))
			print(
				"num_samples {}, loss {:g}, decoder_loss {:g}, bleu_score {:g}, rouge1 {:g}, rouge2 {:g}, rouge3 {:g}, rougel {:g} ".format(
					num_samples, mean_loss, mean_decoder_loss, bleu_score,
					np.mean(rouge1) * 100, np.mean(rouge2) * 100, np.mean(rouge3) * 100, np.mean(rougel) * 100
				))
			return mean_loss, mean_decoder_loss
		
		def _do_test_step(data_set):
			"""
			Evaluates model on a test set
			"""
			num_samples = len(data_set)
			test = num_samples % FLAGS.batch_size
			batch_num = num_samples // FLAGS.batch_size + 1 if test != 0 else num_samples // FLAGS.batch_size
			
			tf_data_set = tf.data.Dataset.from_generator(lambda: data_set, (
				tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
			                                             ).padded_batch(FLAGS.batch_size,
			                                                            padded_shapes=(tf.TensorShape([None, None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([None, None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([None]),
			                                                                           tf.TensorShape([])),
			                                                            padding_values=(
				                                                            0, 0, 0, 0, 0, 0, 0, 0, 0))  # pad index 0
			
			test_iterator = tf_data_set.make_one_shot_iterator()
			test_one_batch = test_iterator.get_next()
			
			losses = []
			decoder_losses = []
			seq2seq_predicts = []
			seq2seq_y_true = []
			for _ in range(batch_num):
				input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch, path_batch, path_lens_batch, decoder_input_x_batch, decoder_output_x_batch, decoder_lens_batch = session.run(
					test_one_batch)
				
				if FLAGS.use_version != 1:
					decoder_input_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_x_batch,
					                                                                      maxlen=FLAGS.max_decoder_steps,
					                                                                      padding="post",
					                                                                      truncating="post",
					                                                                      value=0)
					decoder_output_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_x_batch,
					                                                                       maxlen=FLAGS.max_decoder_steps,
					                                                                       padding="post",
					                                                                       truncating="post",
					                                                                       value=0)
				feed_dict = {model.input_x: input_x_batch,
				             model.input_role: input_role_batch,
				             model.input_sample_lens: input_sample_lens_batch,
				             model.input_sentences_lens: input_sentences_lens_batch,
				
				             model.path: path_batch,
				             model.path_lens: path_lens_batch,
				
				             model.decoder_inputs: decoder_input_x_batch,
				             model.decoder_outputs: decoder_output_x_batch,
				             model.decoder_lengths: decoder_lens_batch,
				             model.dropout_keep_prob: 1.0,
				             model.training: False
				             }
				fetches = [model.loss, model.decoder_loss, model.infer_predicts]
				loss, decoder_loss, batch_seq2seq_predict = session.run(
					fetches=fetches,
					feed_dict=feed_dict)
				losses.append(loss)
				decoder_losses.append(decoder_loss)
				seq2seq_predicts.extend(batch_seq2seq_predict.tolist())
				seq2seq_y_true.extend(decoder_output_x_batch.tolist())
			
			rouge1 = []
			rouge2 = []
			rouge3 = []
			rougel = []
			bleu_scores = []
			for p, g in zip(seq2seq_predicts, seq2seq_y_true):
				gs = _tokenid_to_natural_sentence_ids(g)
				ps = _tokenid_to_natural_sentence_ids(p)
				gs, ps = list(map(lambda x: str(x), gs)), list(map(lambda x: str(x), ps))
				score = nltk.bleu([gs], ps, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4),
				                  smoothing_function=SmoothingFunction().method1)
				bleu_scores.append(score)
				r_scores = cal_rouge([" ".join(ps)], [" ".join(gs)])
				rouge1.append(r_scores["rouge-1"]["f"])
				rouge2.append(r_scores["rouge-2"]["f"])
				rouge3.append(r_scores["rouge-3"]["f"])
				rougel.append(r_scores["rouge-l"]["f"])
			
			mean_loss = np.mean(losses)
			mean_decoder_loss = np.mean(decoder_losses)
			bleu_score = np.mean(bleu_scores) * 100
			logging.info(
				"num_samples {}, loss {:g}, decoder_loss {:g}, bleu_score {:g}, rouge1 {:g}, rouge2 {:g}, rouge3 {:g}, rougel {:g} ".format(
					num_samples, mean_loss, mean_decoder_loss, bleu_score,
					np.mean(rouge1) * 100, np.mean(rouge2) * 100, np.mean(rouge3) * 100, np.mean(rougel) * 100
				))
			print(
				"num_samples {}, loss {:g}, decoder_loss {:g}, bleu_score {:g}, rouge1 {:g}, rouge2 {:g}, rouge3 {:g}, rougel {:g} ".format(
					num_samples, mean_loss, mean_decoder_loss, bleu_score,
					np.mean(rouge1) * 100, np.mean(rouge2) * 100, np.mean(rouge3) * 100, np.mean(rougel) * 100
				))
			return mean_loss, mean_decoder_loss
		
		tf_train_data_set = tf.data.Dataset.from_generator(lambda: train_data_set, (
			tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
		                                                   ).shuffle(len(train_data_set)).repeat(
			FLAGS.num_epochs).padded_batch(FLAGS.batch_size,
		                                   padded_shapes=(
			                                   tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([]),
			                                   tf.TensorShape([None]),
			                                   tf.TensorShape([None, None]), tf.TensorShape([None]),
			                                   tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])),
		                                   padding_values=(0, 0, 0, 0, 0, 0, 0, 0, 0))  # pad index 0
		
		train_iterator = tf_train_data_set.make_one_shot_iterator()
		train_one_batch = train_iterator.get_next()
		
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=FLAGS.num_checkpoints)
		losses = deque([])
		losses_steps = deque([])
		# Generate batches
		for batch_id in tqdm(range(batch_num)):
			
			input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch, path_batch, path_lens_batch, decoder_input_x_batch, decoder_output_x_batch, decoder_lens_batch = session.run(
				train_one_batch)
			try:
				_do_train_step(input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch,
				               path_batch, path_lens_batch, decoder_input_x_batch, decoder_output_x_batch,
				               decoder_lens_batch)
			except Exception as e:
				print(e)
			
			current_step = tf.train.global_step(session, global_step)
			current_lr = session.run(learning_rate)
			if current_step % FLAGS.evaluate_every == 0:
				
				logging.info("\nEvaluation:")
				logging.info("batch_no %d, global_step %d, learning_rate %.5f." % (batch_id, current_step, current_lr))
				logging.info("Train result:")
				_do_valid_step(train_data_set_handout)
				logging.info("dev result:")
				valid_loss, *_ = _do_valid_step(valid_data_set)
				logging.info("Test result:")
				_do_test_step(test_data_set)
				
				early_stop = False
				if len(losses) < FLAGS.num_checkpoints:
					losses.append(valid_loss)
					losses_steps.append(current_step)
				else:
					if losses[0] == min(losses):
						logging.info("early stopping in batch no %d" % batch_id)
						early_stop = True
					else:
						losses.popleft()
						losses.append(valid_loss)
						losses_steps.popleft()
						losses_steps.append(current_step)
				
				if early_stop:
					print(logging.info("early stop, min valid perplexity is %s." % losses))
					print(logging.info("early stop, stopped at step %d." % losses_steps[0]))
			# break
			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(session, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))
		
		logging.info("***************************")
		logging.info("Final result:")
		print("***************************")
		print("Final result:")
		_do_test_step(test_data_set)


def load_data():
	if os.path.exists(FLAGS.train_pkl_file):
		with open(FLAGS.train_pkl_file, 'rb') as pkl_file:
			return pkl.load(pkl_file)
	else:
		vocab = Vocabulary()
		vocab.load(FLAGS.vocab_model_file, keep_words=FLAGS.vocab_size)
		node_vocab = NodeVocab()
		node_vocab.load(FLAGS.node_vocab_file)
		
		df_train = pd.read_csv(FLAGS.train_data_file, sep="\t")
		df_train.fillna(value="", inplace=True)
		print("train:", df_train.shape)
		
		df_dev = pd.read_csv(FLAGS.dev_data_file, sep="\t")
		df_dev.fillna(value="", inplace=True)
		print("dev:", df_dev.shape)
		
		df_test = pd.read_csv(FLAGS.test_data_file, sep="\t")
		df_test.fillna(value="", inplace=True)
		print("test:", df_test.shape)
		
		def _do_vectorize(df):
			df = df.copy()
			df["sentence"] = df["sentence"].map(eval)
			grouped = df.groupby("doc")
			
			sentence_nums = []
			sentence_cut_words = []
			path_cut_nodes = []
			
			sentence_word_ids = []
			path_node_ids = []
			
			sentences_lens = []
			path_lens = []
			
			roles = []
			for agg_name, agg_df in grouped:
				sentence_nums.append(len(agg_df))
				roles.append(agg_df["role"])
				tmp_words = []
				tmp_node = []
				
				for words in agg_df["sentence"]:
					if len(words) <= FLAGS.max_sequence_length:
						tmp_words.append(words)
					else:
						tmp_words.append(words[:FLAGS.max_sequence_length])
				
				for nodes in agg_df["path"]:
					node = nodes.split("->")
					if len(node) <= FLAGS.max_path_length:
						tmp_node.append(node)
					else:
						tmp_node.append(node[:FLAGS.max_path_length])
				
				sentence_cut_words.append(tmp_words)
				path_cut_nodes.append(tmp_node)
				
				sentences_lens.append([len(x) for x in tmp_words])
				path_lens.append([len(y) for y in tmp_node])
				
				word_ids = [vocab.do_encode(x)[0] for x in tmp_words]
				node_ids = [node_vocab.do_encode(y)[0] for y in tmp_node]
				
				word_ids = tf.keras.preprocessing.sequence.pad_sequences(
					word_ids, maxlen=FLAGS.max_sequence_length, padding="post", truncating="post", value=0
				)
				node_ids = tf.keras.preprocessing.sequence.pad_sequences(
					node_ids, maxlen=FLAGS.max_path_length, padding="post", truncating="post", value=0
				)
				assert np.max(word_ids) < FLAGS.vocab_size
				assert np.max(agg_df["role"]) < 6
				
				sentence_word_ids.append(word_ids)
				path_node_ids.append(node_ids)
			
			return sentence_word_ids, roles, sentence_nums, sentences_lens, path_node_ids, path_lens
		
		def _do_label_vectorize(df):
			
			df = df.copy()
			df.index = range(len(df))
			df["sentence"] = df["sentence"].map(eval)
			grouped = df.groupby("doc")
			
			decoder_input_word_ids = []
			decoder_output_word_ids = []
			decoder_sentence_lens = []
			
			for agg_name, agg_df in grouped:
				question = {x for x in agg_df["question"]}
				# question.remove("0")
				question_text = question.pop()
				
				cut_words = eval(question_text)
				# print(cut_words)
				decoder_input_word_ids.append(
					vocab.do_encode(cut_words, mode="bos")[0]
				)
				decoder_output_word_ids.append(
					vocab.do_encode(cut_words, mode="eos")[0]
				)
				decoder_sentence_lens.append(
					len(cut_words) + 1
				)
			
			return decoder_input_word_ids, decoder_output_word_ids, decoder_sentence_lens
		
		# encode x
		train_sentence_word_ids, train_roles, train_sentence_nums, train_sentences_lens, train_path_node_ids, train_path_lens = _do_vectorize(
			df_train)
		dev_sentence_word_ids, dev_roles, dev_sentence_nums, dev_sentences_lens, dev_path_node_ids, dev_path_lens = _do_vectorize(
			df_dev)
		test_sentence_word_ids, test_roles, test_sentence_nums, test_sentences_lens, test_path_node_ids, test_path_lens = _do_vectorize(
			df_test)
		
		train_decoder_input_word_ids, train_decoder_output_word_ids, train_decoder_sentence_lens = _do_label_vectorize(
			df_train)
		dev_decoder_input_word_ids, dev_decoder_output_word_ids, dev_decoder_sentence_lens = _do_label_vectorize(df_dev)
		test_decoder_input_word_ids, test_decoder_output_word_ids, test_decoder_sentence_lens = _do_label_vectorize(
			df_test)
		
		with open(FLAGS.train_pkl_file, 'wb') as pkl_file:
			data = [
				list(zip(train_sentence_word_ids, train_roles, train_sentence_nums, train_sentences_lens,
				         train_path_node_ids, train_path_lens,
				         train_decoder_input_word_ids, train_decoder_output_word_ids,
				         train_decoder_sentence_lens)),
				list(zip(dev_sentence_word_ids, dev_roles, dev_sentence_nums, dev_sentences_lens,
				         dev_path_node_ids, dev_path_lens,
				         dev_decoder_input_word_ids, dev_decoder_output_word_ids,
				         dev_decoder_sentence_lens)),
				list(zip(test_sentence_word_ids, test_roles, test_sentence_nums, test_sentences_lens,
				         test_path_node_ids, test_path_lens,
				         test_decoder_input_word_ids, test_decoder_output_word_ids,
				         test_decoder_sentence_lens))
			]
			pkl.dump(data, pkl_file)
		
		return data


def main(argv=None):
	print("\nParameters:")
	pprint(FLAGS.flag_values_dict())
	
	print("\nLoading data...")
	train_data, valid_data, test_data = load_data()
	
	print("Loading embeddings...")
	FLAGS.pre_trained_word_embeddings = WordTable(FLAGS.word_embedding_file, FLAGS.edim, FLAGS.vocab_size).embeddings
	FLAGS.pre_node_embeddings = NodeTable(FLAGS.node_embedding_file, FLAGS.node_vocab_file, FLAGS.edim).embeddings
	print("embeddings load completed")
	
	_, train_handout_data = train_test_split(train_data, test_size=0.05, random_state=2019)
	
	print("train_data %d, train_data_handout %d, valid_data %d, test_data %d." % (
		len(train_data), len(train_handout_data), len(valid_data), len(test_data)))
	
	data_dict = {
		"train_data_set": train_data,
		"train_data_set_handout": train_handout_data,
		"valid_data_set": valid_data,
		"test_data_set": test_data
	}
	print("\nSampling data...")
	pass
	
	print("\nTraining...")
	train(data_dict)


if __name__ == '__main__':
	tf.app.run()

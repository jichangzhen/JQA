import tensorflow as tf
from ops import gelu, get_shape_list, weight_noise, tensor_noise
import pgn_modeling as pgn_modeling
from bert_modeling import create_attention_mask_from_input_mask, transformer_model, get_assignment_map_from_checkpoint
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, MultiRNNCell, GRUCell, DropoutWrapper
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell, CudnnCompatibleGRUCell

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
Epsilon = 1e-5


class SelfattlSeq2seqBase(object):
	"""
	Language model for Dialogue Act Sequence Labeling using Hierarchical encoder with CRF .
	"""
	
	def __init__(self, config):
		self.config = config
		self.activation_function = {
			"relu": tf.nn.relu,
			"swish": tf.nn.swish,
			"elu": tf.nn.elu,
			"crelu": tf.nn.crelu,
			"tanh": tf.nn.tanh,
			"gelu": gelu
		}[self.config.activation_function]
		
		self.pad_id = 0
		self.go_id = 1
		self.eos_id = 2
		self.unk_id = 3
		self.beam_width = 3
		
		with tf.name_scope("placeholder"):
			
			self.input_x = tf.placeholder(tf.int32, [None, None, None],
			                              name="input_x")  # batch_size, max_sentence_num, max_sequence_length
			self.input_sentences_lens = tf.placeholder(tf.int32, [None, None],
			                                           name="input_sentences_lens")  # batch_size, max_sentence_num
			
			self.input_sample_lens = tf.placeholder(tf.int32, [None], name="input_sample_lens")  # batch_size
			self.input_role = tf.placeholder(tf.int32, [None, None], name="input_role")  # batch_size, max_sentence_num
			
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
			self.training = tf.placeholder(tf.bool, name="bn_training")
			self.input_y = tf.placeholder(tf.int32, [None], name="input_y")  # ground truth
			
			self.path = tf.placeholder(tf.int32, [None, None, None],
			                           name='path')  # batch_size, max_sentence_num, max_path_length
			self.path_lens = tf.placeholder(tf.int32, [None, None], name="path_lens")  # batch_size, max_sentence_num
			
			self.input_sample_mask = tf.sequence_mask(self.input_sample_lens,
			                                          name="input_sample_mask")  # batch_size, max_sentence_num
			self.input_sentences_mask = tf.sequence_mask(self.input_sentences_lens,
			                                             name="input_sentences_mask")  # batch_size, max_sentence_num, max_sequence_length
			self.input_path_mask = tf.sequence_mask(self.path_lens,
			                                        name="input_path_mask")  # batch_size, max_sentence_num, max_path_length
			
			batch_size, max_sentence_num, max_sequence_length = get_shape_list(self.input_x)
			self.batch_size = batch_size
			self.max_sentence_num = max_sentence_num
			self.max_sequence_length = max_sequence_length
			_, _, max_path_length = get_shape_list(self.path)
		
		# Embedding layer with dense
		with tf.name_scope("embedding"):
			with tf.device("/cpu:0"):
				self.word_table = tf.Variable(self.config.pre_trained_word_embeddings, trainable=True, dtype=tf.float32,
				                              name='word_table')
				self.node_table = tf.Variable(self.config.pre_node_embeddings, trainable=True, dtype=tf.float32,
				                              name='node_table')
				
				if self.config.use_role_embedding:
					self.role_table = tf.Variable(tf.truncated_normal([self.config.role_num + 1, self.config.role_edim],
					                                                  stddev=self.config.init_std), trainable=True,
					                              dtype=tf.float32, name='role_table')
					role_embedding = tf.nn.embedding_lookup(self.role_table, self.input_role,
					                                        name='role_embedding')  # (?, ?, 100)
				
				if self.config.use_knowledge:
					nodes_embedding = tf.nn.embedding_lookup(self.node_table, self.path,
					                                         name='nodes_embedding')  # (?, ?, ?, 300)
			
			sample_embedding = tf.nn.embedding_lookup(self.word_table, self.input_x,
			                                          name='sample_embedding')  # (?, ?, ?, 300)
		
		with tf.variable_scope("utterance_rnn"):
			if self.config.use_role_embedding:
				tiled_role_embedding = tf.multiply(
					tf.ones([batch_size, max_sentence_num, max_sequence_length, self.config.role_edim],
					        dtype=tf.float32),
					tf.expand_dims(role_embedding, axis=2)
					# batch_size, max_sentence_num, max_sequence_length, role_embedding(100)
				)  # batch_size, max_sentence_num, max_sequence_length, role_embedding(100)
				
				sample_embedding = tf.concat([sample_embedding, tiled_role_embedding],
				                             axis=-1)  # (batch_size, max_sentence_num, max_sequence_length, 400)
			
			sample_embedding = tf.reshape(sample_embedding, [-1, max_sequence_length, sample_embedding.get_shape()[
				-1].value])  # (batch_size*max_sentence_num, max_sequence_length, 400)
			mask = tf.sequence_mask(tf.reshape(self.input_sentences_lens, [-1]),
			                        maxlen=max_sequence_length)  # (batch_size*max_sentence_num, max_sequence_length)
			mask = tf.cast(tf.expand_dims(mask, axis=-1),
			               dtype=tf.float32)  # (batch_size*max_sentence_num, max_sequence_length, 1)
			sample_embedding = tf.multiply(sample_embedding,
			                               mask)  # (batch_size*max_sentence_num, max_sequence_length, 400)
			
			cell_fw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			cell_bw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			
			(output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=cell_fw, cell_bw=cell_bw, inputs=sample_embedding,
				dtype=tf.float32, sequence_length=tf.reshape(self.input_sentences_lens, [-1])
			)
			
			# print(output_state_fw, output_state_bw)
			final_states = tf.concat([output_state_fw[0].h, output_state_bw[0].h], axis=1)
			utterance_memory_embeddings = tf.concat([output_fw, output_bw], axis=2)
			
			# RNN final state
			sample_text_final_state = tf.reshape(
				tf.concat(final_states, axis=1), [batch_size, max_sentence_num, 2 * self.config.rnn_hidden_size]
			)
			
			# RNN attention
			utterance_memory_embeddings = tf.multiply(utterance_memory_embeddings, mask)
			utterance_memory_embeddings = tf.nn.dropout(utterance_memory_embeddings, keep_prob=self.dropout_keep_prob,
			                                            name="utterance_memory_embeddings")
			
			sample_text_final_state, sen_level_att_score = self.attention_mechanism(utterance_memory_embeddings,
			                                                                        tf.squeeze(mask, axis=-1))
			sample_text_final_state = tf.reshape(sample_text_final_state,
			                                     [batch_size, max_sentence_num, 2 * self.config.rnn_hidden_size])
			
			sen_level_att_score = tf.reshape(sen_level_att_score, [batch_size, max_sentence_num, max_sequence_length])
			self.sen_level_att_score = sen_level_att_score
		
		with tf.variable_scope("node_rnn"):
			nodes_embedding = tf.reshape(nodes_embedding, [-1, max_path_length, nodes_embedding.get_shape()[
				-1].value])  # (batch_size*max_sentence_num, path_len, 300)
			node_mask = tf.sequence_mask(tf.reshape(self.path_lens, [-1]),
			                             maxlen=max_path_length)  # (batch_size*max_sentence_num, path_len)
			node_mask = tf.cast(tf.expand_dims(node_mask, axis=-1),
			                    dtype=tf.float32)  # (batch_size*max_sentence_num, path_len, 1)
			nodes_embedding = tf.multiply(nodes_embedding, node_mask)  # (batch_size*max_sentence_num, path_len, 300)
			
			node_fw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			node_bw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			
			(node_fw, node_bw), (node_state_fw, node_state_bw) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=node_fw, cell_bw=node_bw, inputs=nodes_embedding,
				dtype=tf.float32, sequence_length=tf.reshape(self.path_lens, [-1])
			)
			# print(node_state_fw, node_state_bw) # (?, 200)
			node_final_states = tf.concat([node_state_fw[0].h, node_state_bw[0].h], axis=1)
			node_memory_embeddings = tf.concat([node_fw, node_bw],
			                                   axis=2)  # (batch_size*max_sentence_num, path_len, 400)
			
			# RNN final state
			node_text_final_state = tf.reshape(
				tf.concat(node_final_states, axis=1), [batch_size, max_sentence_num, 2 * self.config.rnn_hidden_size]
			)  # (?, ?, 400)
			
			# RNN attention
			node_memory_embeddings = tf.multiply(node_memory_embeddings, node_mask)
			node_memory_embeddings = tf.nn.dropout(node_memory_embeddings, keep_prob=self.dropout_keep_prob,
			                                       name="node_memory_embeddings")
			
			node_text_final_state, _ = self.attention_mechanism(node_memory_embeddings,
			                                                    tf.squeeze(node_mask, axis=-1))  # (?, 400) # (?, ?, 1)
			node_text_final_state = tf.reshape(node_text_final_state, [batch_size, max_sentence_num,
			                                                           2 * self.config.rnn_hidden_size])  # (batch_size, max_sentence_num, 400)
			
		with tf.variable_scope("utterance_representation"):
			
			if self.config.use_role_embedding:
				final_states = tf.concat([role_embedding, sample_text_final_state], axis=2)
			if self.config.use_knowledge:
				final_states = tf.concat([node_text_final_state, final_states], axis=2)
			else:
				final_states = sample_text_final_state
		
		with tf.variable_scope("enhance_intent"):
			final_states = self.BidirectionalGRUEncoder(final_states)  # (batch_size, max_sentence_num, 400)
			weights_a = tf.get_variable('weights_a', [self.config.rnn_hidden_size * 2, self.config.rnn_hidden_size * 2],
			                            dtype=tf.float32,
			                            initializer=tf.truncated_normal_initializer(stddev=1e-4))
			bias_a = tf.get_variable('bias_a', [self.config.rnn_hidden_size * 2], dtype=tf.float32,
			                         initializer=tf.truncated_normal_initializer(stddev=1e-4))
			weights_c = tf.get_variable('weights_c',
			                            [1, self.config.rnn_hidden_size * 8, self.config.rnn_hidden_size * 2],
			                            dtype=tf.float32,
			                            initializer=tf.truncated_normal_initializer(stddev=1e-4))
			bias_c = tf.get_variable('bias_c', [self.config.rnn_hidden_size * 2], dtype=tf.float32,
			                         initializer=tf.truncated_normal_initializer(stddev=1e-4))
			
			node_state = tf.reshape(node_text_final_state, [-1, 2 * self.config.rnn_hidden_size])
			node_state = tf.matmul(node_state, weights_a) + bias_a
			node_state = tf.tanh(node_state)  # (batch_size*max_sentence_num, 400)
			
			text_pad = tf.expand_dims(node_state, axis=1)
			node_text_state = tf.tile(text_pad, [1, self.config.max_path_length, 1])
			node_text_state, _ = self.attention_mechanism(node_text_state, tf.squeeze(node_mask, axis=-1))
			node_text_state = tf.reshape(node_text_state,
			                             [batch_size, max_sentence_num, 2 * self.config.rnn_hidden_size])
			
			weights_c = tf.tile(weights_c, [tf.shape(final_states)[0], 1, 1])
			final_concat_state = tf.concat(
				[final_states, node_text_state, final_states - node_text_state, final_states * node_text_state],
				axis=-1)
		
		with tf.variable_scope("gating"):
			
			final_state = tf.matmul(final_concat_state, weights_c) + bias_c
			final_state = tf.tanh(final_state)
			mask_node = tf.cast(tf.expand_dims(self.input_sample_mask, axis=-1), dtype=tf.float32)
			final_state = tf.multiply(mask_node, final_state)
			final_state = self.BidirectionalGRUEncoder(final_state)
		
		with tf.variable_scope("dialogue_rnn"):
			mask = tf.cast(tf.expand_dims(self.input_sample_mask, axis=-1), dtype=tf.float32)
			final_states = tf.multiply(mask, final_states)
			
			if self.config.use_version != 3:
				cell_fw = MultiRNNCell(
					[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
					 range(self.config.rnn_layer_num)]
				)
				cell_bw = MultiRNNCell(
					[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
					 range(self.config.rnn_layer_num)]
				)
			
			self.rand_unif_init = tf.random_uniform_initializer(-0.02, 0.02, seed=123)
			self.trunc_norm_init = tf.truncated_normal_initializer(stddev=1e-4)
			
			if self.config.use_version == 3:
				cell_fw = tf.contrib.rnn.LSTMCell(self.config.rnn_hidden_size, initializer=self.rand_unif_init,
				                                  state_is_tuple=True)
				cell_bw = tf.contrib.rnn.LSTMCell(self.config.rnn_hidden_size, initializer=self.rand_unif_init,
				                                  state_is_tuple=True)
			
			(outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=cell_fw, cell_bw=cell_bw, inputs=final_state,
				dtype=tf.float32, sequence_length=self.input_sample_lens,
				swap_memory=True
			)
			
			outputs = tf.concat(outputs, axis=2)
			
			if self.config.use_version == 3:
				self.dec_in_state = self.reduce_states(fw_st, bw_st)
			
			sample_hidden_states = tf.multiply(outputs, mask)
			sample_hidden_states = tf.nn.dropout(sample_hidden_states, keep_prob=self.dropout_keep_prob)
		
		if self.config.transformer_layers > 0:
			with tf.variable_scope("transformer"):
				attention_mask = create_attention_mask_from_input_mask(from_tensor=sample_hidden_states,
				                                                       to_mask=self.input_sample_mask)
				self.all_encoder_layers = transformer_model(
					input_tensor=sample_hidden_states,
					attention_mask=attention_mask,
					hidden_size=self.config.rnn_hidden_size * 2,
					num_hidden_layers=self.config.transformer_layers,
					num_attention_heads=self.config.heads,
					intermediate_size=self.config.intermediate_size,
					intermediate_act_fn=gelu,
					hidden_dropout_prob=1.0 - self.dropout_keep_prob,
					initializer_range=self.config.init_std,
					do_return_all_layers=True)
				
				self.encoder_outputs = self.all_encoder_layers[-1]
				# self.encoder_outputs = tf.nn.dropout(self.encoder_outputs, keep_prob=self.dropout_keep_prob)
		
		with tf.variable_scope("Decoder"):
			
			with tf.variable_scope("decoder_embedding"):
				if self.config.use_version == 3:
					self.decoder_outputs = tf.placeholder(shape=[None, self.config.max_decoder_steps], dtype=tf.int32,
					                                      name="decoder_outputs")
					self.decoder_inputs = tf.placeholder(shape=[None, self.config.max_decoder_steps], dtype=tf.int32,
					                                     name="decoder_inputs")
					self.dec_input = [tf.nn.embedding_lookup(self.word_table, x) for x in
					                  tf.unstack(self.decoder_inputs, axis=1)]
				else:
					self.decoder_outputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_outputs")
					self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs")
				
				self.decoder_lengths = tf.placeholder(shape=[None], dtype=tf.int32, name="decoder_length")
				self.dec_sample_maks = tf.sequence_mask(self.input_sample_lens, dtype=tf.float32,
				                                        name="dec_sample_maks")
				self.prev_coverage = tf.placeholder(tf.float32, [None, None], name='prev_coverage')
				
				self.decoder_emb_inp = tf.nn.embedding_lookup(self.word_table, self.decoder_inputs,
				                                              name="decoder_embeddings")
				self.projection_layer = tf.layers.Dense(self.config.vocab_size, use_bias=True, name="projection_layer")
				self.projection_layer_pointer = tf.layers.Dense(self.config.pointer_vocab_size, use_bias=True,
				                                                name="projection_layer_pointer")
				self.transformer_projection_layer = tf.layers.Dense(self.config.edim, use_bias=True,
				                                                    name="transformer_projection_layer")
				
				self.decoder_cell = CudnnCompatibleLSTMCell(self.config.rnn_hidden_size * 2)
				
				if config.pointer_gen:
					max_word_index = tf.cond(
						self.training,
						lambda: tf.reduce_max([tf.reduce_max(self.input_x), tf.reduce_max(self.decoder_inputs)]),
						lambda: tf.reduce_max(self.input_x)
					)
					self._max_art_oovs = tf.cond(
						max_word_index >= self.config.pointer_vocab_size,
						lambda: max_word_index - self.config.pointer_vocab_size + 1,
						lambda: 0
					)
			
			if self.config.use_version == 1:
				with tf.variable_scope("attention_layer_v1"):
					attention_mechanism = tf.contrib.seq2seq.LuongAttention(
						self.config.rnn_hidden_size * 2,
						memory=self.encoder_outputs,
						memory_sequence_length=self.input_sample_lens,
						scale=True)
					
					self.decoder_emb_inp = tf.transpose(self.decoder_emb_inp, [1, 0, 2])
					self.decoder_emb_inp = tensor_noise(self.decoder_emb_inp, self.config.input_noise_std,
					                                    self.training)
					self.decoder_emb_inp = tf.nn.dropout(self.decoder_emb_inp, self.dropout_keep_prob)
					
					self.decoder_cell_wrapper = seq2seq.AttentionWrapper(
						self.decoder_cell, attention_mechanism, alignment_history=True,
						attention_layer_size=self.config.rnn_hidden_size * 2)
				
				with tf.variable_scope("attention_decoder_v1"):
					# Helper
					train_helper = seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_lengths, time_major=True)
					
					self.decoder_initial_state = self.decoder_cell_wrapper.zero_state(tf.shape(self.input_x)[0],
					                                                                  tf.float32)
					
					# Decoder
					train_decoder = seq2seq.BasicDecoder(
						self.decoder_cell_wrapper, train_helper, self.decoder_initial_state,  # encoder_state
						output_layer=self.projection_layer
					)
					
					self.train_outputs, _, _ = seq2seq.dynamic_decode(
						train_decoder,
						output_time_major=True,
						swap_memory=True,
					)
			
			if self.config.use_version == 2:
				with tf.variable_scope("attention_layer_v2"):
					
					attention_mechanism = pgn_modeling.MyLuongAttention(
						2 * self.config.rnn_hidden_size,
						memory=self.encoder_outputs,
						memory_sequence_length=self.input_sample_lens,
						scale=True)
					
					self.decoder_emb_inp = tf.transpose(self.decoder_emb_inp, [1, 0, 2])
					self.decoder_emb_inp = tensor_noise(self.decoder_emb_inp, self.config.input_noise_std,
					                                    self.training)
					# self.decoder_emb_inp = tf.nn.dropout(self.decoder_emb_inp, self.dropout_keep_prob)
					
					self.decoder_cell_wrapper = pgn_modeling.MyAttentionWrapper(
						self.decoder_cell, attention_mechanism,
						attention_layer_size=self.config.rnn_hidden_size * 2,
						name="attention_wrapper"
					)
				
				with tf.variable_scope("attention_decoder_v2"):
					train_helper = seq2seq.TrainingHelper(
						self.decoder_emb_inp,
						tf.ones(shape=(self.batch_size,), dtype=tf.int32) * self.config.max_decoder_steps,
						time_major=True
					)
					
					# train_helper = seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_lengths, time_major=True)
					# Returned sample_ids are the argmax of the RNN output logits
					
					self.decoder_initial_state = self.decoder_cell_wrapper.zero_state(self.batch_size, tf.float32)
					# self.decoder_initial_state = self.decoder_cell_wrapper.zero_state(tf.shape(self.input_x)[0], tf.float32)
					# 初始化全零 state _init_state会生成一个元组(c_state,m_state)
					
					train_decoder = pgn_modeling.MyBasicDecoder(
						self.decoder_cell_wrapper, train_helper, self.decoder_initial_state,  # encoder_state
					)
					
					self.train_outputs, _, _, attn_dists, seq_inputs = pgn_modeling.my_dynamic_decode(
						train_decoder,
						output_time_major=True,
						swap_memory=True,
					)  # train_outputs(? ,64, 400) (?, 64)  attn_dists(?, ?, ?) seq_inputs(?, ?, 300)
					
					if self.config.use_transformer_linear_projection:
						# max_decoder_steps, batch_size, edim
						dec = self.transformer_projection_layer(
							tf.nn.dropout(self.train_outputs.rnn_output, keep_prob=self.dropout_keep_prob))
						
						# batch_size, max_decoder_steps, edim
						dec = tf.transpose(dec, [1, 0, 2])
						weights = tf.transpose(self.word_table)  # (300, 20001)
						self.seq2seq_logits = tf.einsum('ntd,dk->ntk', dec,
						                                weights)  # (N, T2, vocab_size) #(?, ?, 20001)
					
					else:
						self.seq2seq_logits = self.projection_layer_pointer(
							tf.nn.dropout(self.train_outputs.rnn_output, keep_prob=self.dropout_keep_prob))
						self.seq2seq_logits = tf.transpose(self.seq2seq_logits, [1, 0, 2])  # (?, ?, 20001)
					
					vocab_dists = tf.nn.softmax(self.seq2seq_logits)  # (64, ?, 20001)
					
					if self.config.pointer_gen:
						vocab_dists.set_shape(
							shape=[None, self.config.max_decoder_steps, self.config.pointer_vocab_size])
					else:
						vocab_dists.set_shape(shape=[None, self.config.max_decoder_steps, self.config.vocab_size])
					
					attn_dists.set_shape(shape=[None, None, self.config.max_decoder_steps])
					self.vocab_dists = tf.unstack(vocab_dists, axis=1)  # max_decoder_steps (?, 20001)
					attn_dists = tf.unstack(attn_dists, axis=2)  # max_decoder_steps (?, ?)
					
					self.attn_dists = []
					for dist in attn_dists:
						attn_dist = tf.multiply(tf.expand_dims(dist, axis=-1), sen_level_att_score)
						self.attn_dists.append(attn_dist)  # batch, max_sen_num, max_sen_len
					
					self.p_gen_dense_rnn = tf.layers.Dense(1, use_bias=False)
					self.p_gen_dense_input = tf.layers.Dense(1, use_bias=True)
					
					# seq_inputs = tf.transpose(seq_inputs, [1, 0, 2])  # timestep, batch_size, input_size
					
					self.p_gens = tf.nn.sigmoid(
						self.p_gen_dense_rnn(self.train_outputs.rnn_output) +
						self.p_gen_dense_input(self.decoder_emb_inp)
					)  # self.train_outputs.rnn_output(? 64 400) self.decoder_emb_inp(100, 64, 300) self.p_gens(100, 64, 1)
					
					self.p_gens = tf.squeeze(self.p_gens, axis=2)
					self.p_gens.set_shape([self.config.max_decoder_steps, None])
					self.p_gens = tf.unstack(self.p_gens, axis=0)  # max_decoder_steps (64, ?)
					
					if self.config.pointer_gen:
						self.final_dists = self._calc_final_dist_v1(self.vocab_dists, self.attn_dists, self.p_gens)
					else:  # final distribution is just vocabulary distribution
						self.final_dists = self.vocab_dists  # max_decoder_steps (batch, 20001)
					
					self.seq2seq_predicts = tf.argmax(tf.stack(self.final_dists, axis=1),
					                                  axis=2)  # tf.stack->(?, 100, 20001) # (?, 100)
			
			if self.config.use_version == 3:
				with tf.variable_scope("attention_layer_v3"):
					
					attention_mechanism = pgn_modeling.MyLuongAttention(
						2 * self.config.rnn_hidden_size,
						memory=self.encoder_outputs,
						memory_sequence_length=self.input_sample_lens,
						scale=True)
					
					self.decoder_emb_inp = tf.transpose(self.decoder_emb_inp, [1, 0, 2])
					self.decoder_emb_inp = tensor_noise(self.decoder_emb_inp, self.config.input_noise_std,
					                                    self.training)
					self.decoder_emb_inp = tf.nn.dropout(self.decoder_emb_inp, self.dropout_keep_prob)
					
					self.decoder_cell_wrapper = pgn_modeling.MyAttentionWrapper(
						self.decoder_cell, attention_mechanism,
						attention_layer_size=self.config.rnn_hidden_size * 2,
						name="attention_wrapper"
					)
					
					self.p_gen_dense_rnn = tf.layers.Dense(1, use_bias=False)
					self.p_gen_dense_input = tf.layers.Dense(1, use_bias=True)
				
				with tf.variable_scope("attention_decoder_v3"):
					self.cell = tf.contrib.rnn.LSTMCell(self.config.rnn_hidden_size, state_is_tuple=True,
					                                    initializer=self.rand_unif_init)
					outputs, self.out_state, self.attn_dists, self.p_gens, self.coverage = pgn_modeling.attention_decoder(
						self.dec_input,
						self.dec_in_state,
						self.encoder_outputs,
						self.dec_sample_maks,
						self.cell,
						initial_state_attention=False,
						pointer_gen=self.config.pointer_gen,
						use_coverage=self.config.coverage,
						prev_coverage=None)
				
				with tf.variable_scope('output_projection'):
					hidden_dim = self.config.rnn_hidden_size
					vsize = self.config.vocab_size
					w = tf.get_variable('w', [hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
					w_t = tf.transpose(w)
					v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
					vocab_scores = []
					for i, output in enumerate(outputs):
						if i > 0:
							tf.get_variable_scope().reuse_variables()
						vocab_scores.append(tf.nn.xw_plus_b(output, w, v))
					
					self.vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
					
					if self.config.pointer_gen:
						self.final_dists = self._calc_final_dist_v2(self.vocab_dists, self.attn_dists)
					else:  # final distribution is just vocabulary distribution
						self.final_dists = self.vocab_dists  # max_decoder_steps (batch, 20001)
			
			if self.config.use_version == 1:
				with tf.variable_scope("decoder_loss_v1"):
					self.logits = self.train_outputs.rnn_output
					self.predicts = self.train_outputs.sample_id
					self.logits = tf.transpose(self.logits, [1, 0, 2])
					
					target_weights = tf.sequence_mask(self.decoder_lengths, dtype=tf.float32)
					self.decoder_loss = seq2seq.sequence_loss(logits=self.logits, targets=self.decoder_outputs,
					                                          weights=target_weights)
					self.loss = self.decoder_loss
			
			else:
				with tf.variable_scope("decoder_loss_v2"):
					if self.config.pointer_gen:
						loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
						batch_nums = tf.range(0, limit=self.batch_size)  # shape (batch_size)
						for dec_step, dist in enumerate(self.final_dists):
							targets = self.decoder_outputs[:,
							          dec_step]  # The indices of the target words. shape (batch_size)
							indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
							gold_probs = tf.gather_nd(dist,
							                          indices)  # shape (batch_size). prob of correct words on this step
							gold_probs = tf.clip_by_value(gold_probs, 1e-6, 1 - 1e-6)
							losses = -tf.log(gold_probs)
							loss_per_step.append(losses)
						
						self.loss_per_step = tf.stack(loss_per_step, axis=1)  # batch_size, max_decoder_len
						target_weights = tf.sequence_mask(self.decoder_lengths, dtype=tf.float32,
						                                  maxlen=self.config.max_decoder_steps)
						loss = tf.reduce_sum(self.loss_per_step * target_weights, axis=1)
						loss /= tf.cast(self.decoder_lengths, tf.float32)
						self.decoder_loss = tf.reduce_mean(loss)
					
					else:  # baseline model
						target_weights = tf.sequence_mask(self.decoder_lengths, dtype=tf.float32,
						                                  maxlen=self.config.max_decoder_steps)
						self.decoder_loss = seq2seq.sequence_loss(self.seq2seq_logits, self.decoder_outputs,
						                                          target_weights)  # this applies softmax internally
					
					self.regularization_loss = tf.losses.get_regularization_loss()
					self.loss = self.decoder_loss + self.regularization_loss
			
			with tf.variable_scope("infer_decoder"):
				if self.config.use_version == 1:
					self.build_greedy_inference_model_v1()
				else:
					self.build_greedy_inference_model_v2()
		
		tvars = tf.trainable_variables()
		initialized_variable_names = {}
		
		if self.config.fine_tuning:
			(assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars,
			                                                                                  self.config.pre_train_lm_checkpoint_path)
			tf.train.init_from_checkpoint(self.config.pre_train_lm_checkpoint_path, assignment_map)
			
			print("load bert check point done")
		
		tf.logging.info("**** Trainable Variables ****")
		for var in tvars:
			init_string = ""
			if var.name in initialized_variable_names:
				init_string = ", *INIT_FROM_CKPT*"
			
			tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
			print("  name = %s, shape = %s%s", var.name, var.shape, init_string)
	
	@staticmethod
	def attention_mechanism(inputs, x_mask=None):
		"""
		Attention mechanism layer.

		:param inputs: outputs of RNN/Bi-RNN layer (not final state)
		:param x_mask:
		:return: outputs of the passed RNN/Bi-RNN reduced with attention vector
		"""
		# In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
		if isinstance(inputs, tuple):
			inputs = tf.concat(inputs, 2)
		_, sequence_length, hidden_size = get_shape_list(inputs)
		
		v = tf.layers.dense(
			inputs, hidden_size,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
			activation=tf.tanh,
			use_bias=True
		)
		att_score = tf.layers.dense(
			v, 1,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
			use_bias=False
		)  # batch_size, sequence_length, 1
		
		att_score = tf.squeeze(att_score, axis=-1) * x_mask + VERY_NEGATIVE_NUMBER * (
				1 - x_mask)  # [batch_size, sentence_length
		att_score = tf.expand_dims(tf.nn.softmax(att_score), axis=-1)  # [batch_size, sentence_length, 1]
		att_pool_vec = tf.matmul(tf.transpose(att_score, [0, 2, 1]), inputs)  # [batch_size,  h]
		att_pool_vec = tf.squeeze(att_pool_vec, axis=1)
		
		return att_pool_vec, att_score
	
	def BidirectionalGRUEncoder(self, inputs):
		GRU_cell_fw = GRUCell(self.config.rnn_hidden_size)
		GRU_cell_bw = GRUCell(self.config.rnn_hidden_size)
		
		((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
		                                                                     cell_bw=GRU_cell_bw,
		                                                                     inputs=inputs,
		                                                                     sequence_length=self.length(inputs),
		                                                                     dtype=tf.float32)
		# outputs的size是[batch_size, max_time, hidden_size*2]
		outputs = tf.concat((fw_outputs, bw_outputs), 2)
		return outputs
	
	def length(self, sequences):
		used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
		seq_len = tf.reduce_sum(used, reduction_indices=1)
		return tf.cast(seq_len, tf.int32)
	
	def reduce_states(self, fw_st, bw_st):
		hidden_dim = self.config.rnn_hidden_size
		with tf.variable_scope('reduce_final_st'):
			w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
			                             initializer=self.trunc_norm_init)
			w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
			                             initializer=self.trunc_norm_init)
			bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
			                                initializer=self.trunc_norm_init)
			bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
			                                initializer=self.trunc_norm_init)
			
			# Apply linear layer
			old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
			old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
			new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
			# Get new cell from old cell
			new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
			# Get new state from old state
			return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state
	
	def _calc_final_dist_v1(self, vocab_dists, attn_dists, p_gens):
		"""Calculate the final distribution, for the pointer-generator model
		Args:
		  vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
		  attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
		Returns:
		  final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
		"""
		with tf.variable_scope('final_distribution'):
			# Multiply vocab dists by p_gen and attention dists by (1-p_gen)
			vocab_dists = [tf.expand_dims(p_gen, axis=-1) * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
			attn_dists = [tf.expand_dims(tf.expand_dims((1 - p_gen), axis=-1), axis=-1) * dist for (p_gen, dist) in
			              zip(p_gens, attn_dists)]
			# Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
			extended_vsize = self.config.pointer_vocab_size + self._max_art_oovs  # the maximum (over the batch) size of the extended vocabulary
			extra_zeros = tf.zeros((self.batch_size, self._max_art_oovs))
			vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
			                        vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)
			
			# Project the values in the attention distributions onto the appropriate entries in the final distributions
			# This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
			# This is done for each decoder timestep.
			# This is fiddly; we use tf.scatter_nd to do the projection
			batch_nums = tf.range(0, limit=self.batch_size)  # shape (batch_size)
			batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
			# attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
			attn_len = self.max_sequence_length * self.max_sentence_num
			batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
			indices = tf.stack(
				(
					batch_nums,
					tf.reshape(self.input_x, [self.batch_size, self.max_sequence_length * self.max_sentence_num])
				), axis=2
			)  # shape (batch_size, enc_t, 2)
			shape = [self.batch_size, extended_vsize]
			attn_dists_projected = [
				tf.scatter_nd(
					indices,
					tf.reshape(copy_dist, [self.batch_size, self.max_sequence_length * self.max_sentence_num]),
					shape
				) for copy_dist in attn_dists
			]  # list length max_dec_steps (batch_size, extended_vsize)
			
			# Add the vocab distributions and the copy distributions together to get the final distributions
			# final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
			# Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
			final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
			               zip(vocab_dists_extended, attn_dists_projected)]
			
			return final_dists
	
	def _calc_final_dist_v2(self, vocab_dists, attn_dists):
		with tf.variable_scope('final_distribution'):
			vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
			attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]
			extended_vsize = self.config.pointer_vocab_size + self._max_art_oovs
			extra_zeros = tf.zeros((self.batch_size, self._max_art_oovs))
			vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]
			
			batch_nums = tf.range(0, limit=self.batch_size)  # shape (batch_size)
			batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
			attn_len = self.max_sequence_length * self.max_sentence_num  # number of states we attend over
			batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
			indices = tf.stack((batch_nums, tf.reshape(self.input_x, [self.batch_size,
			                                                          self.max_sequence_length * self.max_sentence_num])),
			                   axis=2)
			
			shape = [self.batch_size, extended_vsize]
			attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]
			
			final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
			               zip(vocab_dists_extended, attn_dists_projected)]
			
			return final_dists
	
	def build_greedy_inference_model_v1(self):
		
		# infer Helper
		infer_helper = seq2seq.GreedyEmbeddingHelper(self.word_table, tf.fill([tf.shape(self.input_x)[0]], self.go_id),
		                                             self.eos_id)
		
		# Decoder
		infer_decoder = seq2seq.BasicDecoder(self.decoder_cell_wrapper, infer_helper, self.decoder_initial_state,
		                                     output_layer=self.projection_layer)
		
		maximum_iterations = tf.round(50 * 2)  # max 50 * words outputs
		
		# Dynamic decoding
		self.infer_outputs, _, _ = seq2seq.dynamic_decode(infer_decoder, maximum_iterations=maximum_iterations)
		
		self.infer_predicts = self.infer_outputs.sample_id
	
	def build_greedy_inference_model_v2(self):
		infer_predicts = []
		next_decoder_state = self.decoder_cell_wrapper.zero_state(self.batch_size, dtype=tf.float32)
		next_inputs = tf.nn.embedding_lookup(self.word_table, ids=tf.fill([self.batch_size], self.go_id))
		i = 0
		
		while i < self.config.max_decoder_steps:
			cell_outputs, next_decoder_state, cell_score = self.decoder_cell_wrapper(next_inputs, next_decoder_state)
			
			if self.config.use_transformer_linear_projection:
				# max_decoder_steps, batch_size, edim
				dec = self.transformer_projection_layer(cell_outputs)
				weights = tf.transpose(self.word_table)
				infer_seq2seq_logits = tf.matmul(dec, weights)  # (N, vocab_size)
			else:
				infer_seq2seq_logits = self.projection_layer_pointer(cell_outputs)
			
			infer_vocab_dists = tf.nn.softmax(infer_seq2seq_logits)
			infer_vocab_dists = tf.expand_dims(infer_vocab_dists, axis=1)
			cell_score = tf.expand_dims(cell_score, axis=2)
			infer_vocab_dists = tf.unstack(infer_vocab_dists, axis=1)
			infer_attn_dists = tf.unstack(cell_score, axis=2)
			
			re_weighted_infer_attn_dists = []
			for dist in infer_attn_dists:
				attn_dist = tf.multiply(tf.expand_dims(dist, axis=-1), self.sen_level_att_score)
				re_weighted_infer_attn_dists.append(attn_dist)  # batch, max_sen_num, max_sen_len
			
			infer_p_gens = tf.nn.sigmoid(self.p_gen_dense_rnn(cell_outputs) + self.p_gen_dense_input(next_inputs))
			infer_p_gens = tf.expand_dims(infer_p_gens, axis=0)
			infer_p_gens = tf.squeeze(infer_p_gens, axis=2)
			infer_p_gens.set_shape([1, None])
			infer_p_gens = tf.unstack(infer_p_gens, axis=0)
			
			if self.config.pointer_gen:
				infer_final_dists = self._calc_final_dist_v1(infer_vocab_dists, re_weighted_infer_attn_dists,
				                                             infer_p_gens)
			
			else:  # final distribution is just vocabulary distribution
				infer_final_dists = infer_vocab_dists
			
			infer_final_dists = tf.stack(infer_final_dists, axis=1)
			sample_id = tf.squeeze(tf.argmax(infer_final_dists, axis=2), axis=1)
			next_inputs = tf.nn.embedding_lookup(self.word_table, ids=sample_id)
			infer_predicts.append(sample_id)
			i += 1
		
		self.infer_predicts = tf.stack(infer_predicts, axis=1, name="infer_predicts")
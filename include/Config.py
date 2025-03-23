import tensorflow as tf


class Config:
	language = 'j3_en' # zh_en | ja_en | fr_en
	e1 = 'data/' + language + '/ent_ids_1'
	e2 = 'data/' + language + '/ent_ids_2'
	ill = 'data/' + language + '/sup_ent_ids'
	train = 'data/' + language + '/train_ent_ids'
	test = 'data/' + language + '/test_little'
	val = 'data/' + language + '/val_little'
	kg1 = 'data/' + language + '/triples_1'
	kg2 = 'data/' + language + '/triples_2'
	# X_pool = 'data/' + language + '/test_file'
	X_pool = 'data/' + language + '/entity_pairs_8000'
	model_path = 'include/model/checkpoints'
	epochs = 600
	dim = 300
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.1
	gamma = 1.0  # margin based loss
	k = 125  # number of negative samples for each positive one  基模型
	# k = 1

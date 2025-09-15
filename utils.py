import timeit

import dgl
import numpy as np
import torch
import random
import pickle
import sys

from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity



def set_seeds(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def get_vocab_idx(vocab, token):
	if token not in vocab:
		return vocab['UNK']
	else:
		return vocab[token]

def get_relation_id(rel_name, train_network):
	rel_id = train_network.graph.find_relation(rel_name)
	if rel_id == -1:
		return len(train_network.rel2id)
	else:
		return rel_id


def prepare_batch_dgl(vocab, test_network, train_network):
	all_edges = []
	all_labels = []
	for edge in test_network.graph.iter_edges():
		src_id = get_vocab_idx(vocab, edge.src.name)
		tgt_id = get_vocab_idx(vocab, edge.tgt.name)
		rel_id = get_relation_id(edge.relation.name, train_network)
		all_edges.append(np.array([src_id, rel_id, tgt_id]))
		all_labels.append(edge.label)
	return np.array(all_edges), all_labels


def get_adj_and_degree(num_nodes, num_rels, triplets):
	col = []
	row = []
	rel = []
	adj_list = [[] for _ in range(num_nodes)]
	for i, triplet in enumerate(triplets):
		adj_list[triplet[0]].append([i, triplet[2]])
		adj_list[triplet[2]].append([i, triplet[0]])

		row.append(triplet[0])
		col.append(triplet[2])
		rel.append(triplet[1])
		row.append(triplet[2])
		col.append(triplet[0])
		rel.append(triplet[1] + num_rels)

	sparse_adj_matrix = coo_matrix((np.ones(len(triplets)*2),(row,col)),shape=(num_nodes, num_nodes))
	degrees = np.array([len(a) for a in adj_list])
	adj_list = [np.array(a) for a in adj_list]
	return adj_list, degrees, sparse_adj_matrix, rel


def create_entity_dicts(args, all_tuples, num_rels, sim_relations=False):
	e1_to_multi_e2 ={}
	e2_to_multi_e1 ={}

	for tup in all_tuples:
		e1, rel, e2 = tup


		if rel == num_rels-1 and sim_relations:
			continue

		rel_offset = num_rels

		if sim_relations:
			rel_offset -= 1

		if (e1,rel) in e1_to_multi_e2:
			e1_to_multi_e2[(e1,rel)].append(e2)
		else:
			e1_to_multi_e2[(e1,rel)] = [e2]

		if args.inverse_relation:
			if (e2, rel + rel_offset) in e2_to_multi_e1:
				e2_to_multi_e1[(e2, rel + rel_offset)].append(e1)
			else:
				e2_to_multi_e1[(e2, rel + rel_offset)] = [e1]

	return e1_to_multi_e2, e2_to_multi_e1

def load_pre_computed_feat(feat_path, feat_dim, id2node , logger = None):
	with open(feat_path, 'rb') as fp:
		node_emb_dict = pickle.load(fp)
	load_dim = node_emb_dict[random.choice(list(node_emb_dict))].shape[0]


	node_embed = np.zeros((len(id2node), load_dim))

	for i in range(len(id2node)):
		if id2node[i] in node_emb_dict:
			node_embed[i] = node_emb_dict[id2node[i]]
		elif id2node[i].lower() in node_emb_dict:
			node_embed[i] = node_emb_dict[id2node[i].lower()]
		else:
			logger.info("The embedding of ({}) not in pretrain feature".format(id2node[i]))

	logger.info("Load feature from: {}".format(feat_path))
	logger.info("Load feature shape: {}".format(node_embed.shape))


	if feat_dim>node_embed.shape[1]:
		logger.info("Desired dimension larger than loaded embedding. Please check!")
		sys.exit()
	elif feat_dim<node_embed.shape[1]:
		logger.info("BERT desired dimension smaller than loaded embedding. Perform PCA...")
		pca = PCA(n_components=feat_dim)
		node_embed = pca.fit_transform(node_embed)
		logger.info("New Embedding Shape: {}".format(node_embed.shape))

	node_embed = torch.tensor(node_embed)

	return node_embed


def create_triplet_graph(args, train_data, logger= None):
	num_nodes = args.num_nodes
	num_rels = args.num_rels

	edge_src = []
	tri_edge_type = []
	edge_tgt = []

	for tup in train_data:
		e1, rel, e2 = tup
		edge_src.append(e1)
		edge_tgt.append(e2)
		tri_edge_type.append(rel)

		if args.inverse_relation:
			edge_src.append(e2)
			edge_tgt.append(e1)
			tri_edge_type.append(rel + args.num_rels)

	logger.info("Triplet graph edges: {}".format(len(edge_src)))
	return edge_src, edge_tgt, tri_edge_type



def dynamic_graph_gen(args, entity_embedding, n_ontology=1, inductive_index=[], degrees=[], logger= None):

	if n_ontology < 1:
		logger.info("*********************************************")
		logger.info("Perform global thresholding for graph generation")
		start_time = timeit.default_timer()

		threshold = n_ontology
		num_nodes = args.num_nodes

		sim_edge_src = []
		sim_edge_tgt = []
		sim_edge_type = []

		batch_size = 1000

		for row_i in range(0, int(entity_embedding.shape[0] / batch_size)+1):
			start = row_i * batch_size
			end = min([(row_i + 1)*batch_size, entity_embedding.shape[0]])
			if end < start:
				break
			rows = entity_embedding[start:end]
			sim = cosine_similarity(rows, entity_embedding)

			for i in range(end - start):
				ind = i+start
				sim[i, ind] = 0


			sim_edge_src.extend((np.where(sim >= threshold)[0] + start).tolist())
			sim_edge_tgt.extend((np.where(sim >= threshold)[1]).tolist())



		sim_edge_type = [args.num_edge_types-1]*len(sim_edge_src)
		logger.info("Number of semantic similarity edges: {}".format(len(sim_edge_src)))
		stop_time = timeit.default_timer()
		logger.info("Time: {}".format(stop_time-start_time))
		logger("****************************************")

	else:
		logger.info("*****************************************")
		logger.info("knn graph for graph generation")
		start_time = timeit.default_timer()

		threshold = int(n_ontology)
		num_nodes = args.num_nodes

		sim_edge_src = []
		sim_edge_tgt = []
		sim_edge_type = []

		batch_size = 1000

		for row_i in range(0, int(entity_embedding.shape[0]/batch_size)+1):
			start  = row_i*batch_size
			end = min([(row_i+1)*batch_size, entity_embedding.shape[0]])
			if end <= start:
				break
			rows = entity_embedding[start:end]
			sim = cosine_similarity(rows, entity_embedding)

			for i in range(end -start):
				ind = i + start
				sim[i, ind] = 0

			for i in range(sim.shape[0]):
				if (threshold - degrees[i+start]) <= 0:
					continue
				indexing = np.argsort(sim[i])[-(threshold-degrees[i+start]):]
				for j in range(indexing.shape[0]):
					src = indexing[j]
					sim_edge_src.append(src)
					sim_edge_tgt.append(i+start)




		sim_edge_type = [args.num_edge_types-1] * len(sim_edge_src)
		logger.info('Number of semantic similarity edges: {}'.format(len(sim_edge_src)))
		stop_time = timeit.default_timer()
		logger.info('Time: {}'.format(stop_time - start_time))
		logger.info('**************************')

	logger.info('Number of NA need to be filtered: {}'.format(len(inductive_index)))

	filtered_sim_edge_src = []
	filtered_sim_edge_tgt = []
	filtered_sim_edge_type = []

	for i in range(len(sim_edge_src)):
		if sim_edge_src[i] not in inductive_index and sim_edge_tgt[i] not in inductive_index:
			filtered_sim_edge_src.append(sim_edge_src[i])
			filtered_sim_edge_tgt.append(sim_edge_tgt[i])
			filtered_sim_edge_type.append(sim_edge_type[i])

	logger.info('Number of similarity edges after filtering: {}'.format(len(filtered_sim_edge_src)))
	logger.info('**************************')

	return filtered_sim_edge_src, filtered_sim_edge_tgt, filtered_sim_edge_type


def sample_sub_graph(args, sample_size, tri_graph, logger=None, test_graph_bool=False):

	num_nodes = args.num_nodes
	if sample_size >= num_nodes:
		sample_size = num_nodes

	choice_uniq_v = np.random.choice(num_nodes, sample_size, replace=False)
	choice_uniq_v = np.sort(choice_uniq_v)

	temp_src = []
	temp_dst = []
	temp_edge_type = []

	tri_edge_src, tri_edge_tgt, tri_edge_type = tri_graph

	filtered_edges = []
	for i in range(len(tri_edge_src)):
		head = tri_edge_src[i]
		rel = tri_edge_type[i]
		tail = tri_edge_tgt[i]
		if head in choice_uniq_v and tail in choice_uniq_v:
			temp_src.append(head)
			temp_dst.append(tail)
			temp_edge_type.append(rel)


	edge_type = np.array(temp_edge_type)



	if test_graph_bool:
		src = temp_src
		dst = temp_dst
		uniq_v = choice_uniq_v
	else:
		train_uniq_v, edges = np.unique((temp_src, temp_dst), return_inverse=True)
		src, dst = np.reshape(edges, (2, -1))
		uniq_v = train_uniq_v


	g = dgl.DGLGraph()
	g.add_nodes(len(uniq_v))


	if test_graph_bool:
		g.add_edges(src, dst)
	else:




		num_edges = dst.shape[0]


		keep_ratio = args.keep_true_adjacency
		num_keep = int(float(num_edges) * keep_ratio)
		num_rand = num_edges - num_keep


		indices = np.arange(num_edges)
		np.random.shuffle(indices)


		keep_indices = indices[:num_keep]

		rand_indices = indices[num_keep:]

		mixed_dst = np.empty_like(dst)


		mixed_dst[keep_indices] = dst[keep_indices]


		mixed_dst[rand_indices] = np.random.randint(
			low=0, high=len(uniq_v), size=num_rand
		)


		g.add_edges(src, mixed_dst)




	norm = comp_deg_norm(g)
	logger.info("Nodes_Number:{}, Edges_Number:{}".format(len(uniq_v), len(src)))


	node_id = uniq_v
	node_norm = norm

	node_id_copy = np.copy(node_id)



	node_id = torch.from_numpy(node_id).view(-1,1)
	node_norm = torch.from_numpy(node_norm).view(-1,1)
	edge_type = torch.from_numpy(edge_type)

	node_id = node_id.to(args.device)
	node_norm = node_norm.to(args.device)
	edge_type = edge_type.to(args.device)

	g = g.to(args.device)

	g.ndata.update({'id':node_id, 'norm':node_norm})
	g.edata['type'] = edge_type

	print("choice_uniq_v:{}, uniq_v:{}".format(len(choice_uniq_v), len(uniq_v)))

	return  g, uniq_v, norm


def comp_deg_norm(g):
	in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
	norm = 1.0/in_deg
	norm[np.isinf(norm)] = 0
	return norm
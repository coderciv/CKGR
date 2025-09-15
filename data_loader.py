from graph_class import Graph
import utils


from collections import Counter
import os
import numpy as np





class Reader_Data:

	def __init__(self, dataset, logger):
		self.dataset = dataset
		self.graph = Graph(logger= logger)
		self.rel2id = {}


	def print_summary(self, logger = None):
		logger.info("Graph Summary")
		logger.info("Nodes: {}".format(len(self.graph.nodes)))
		logger.info("Edges: {}".format(self.graph.edgeCount))
		logger.info("Relations: {}".format(len(self.graph.relations2id)))
		density = self.graph.edgeCount / (len(self.graph.nodes) * (len(self.graph.nodes)-1))
		logger.info("Density: {}".format(density))

		for i,edge in enumerate(self.graph.iter_edges()):
			logger.info("{}".format(edge))
			if (i+1) % 10 == 0:
				break

	def add_example(self, src, tgt, relation, weight, label=1, train_network=None):
		src_id = self.graph.find_node(src)
		if src_id == -1:
			src_id = self.graph.add_node(src)

		tgt_id = self.graph.find_node(tgt)
		if tgt_id == -1:
			tgt_id = self.graph.add_node(tgt)

		relation_id = self.graph.find_relation(relation)
		if relation_id == -1:
			relation_id = self.graph.add_relation(relation)


		edge = self.graph.add_edge(self.graph.nodes[src_id],
		                           self.graph.nodes[tgt_id],
		                           self.graph.relations[relation_id],
		                           label,
		                           weight)


		new_added = 0
		if train_network is not None and label == 1:
			src_id = train_network.graph.find_node(src)
			if src_id == -1:
				src_id = train_network.graph.add_node(src)
				new_added += 1

			tgt_id = train_network.graph.find_node(tgt)
			if tgt_id == -1:
				tgt_id = train_network.graph.add_node(tgt)
				new_added += 1

			relation_id = train_network.graph.find_relation(relation)
			if relation_id == -1:
				relation_id = train_network.graph.add_relation(relation)

		return edge, new_added


	def read_data(self, data_dir, split="train", train_network=None, logger = None):
		if split == "train":
			data_path = os.path.join(data_dir, "train.txt")
		elif split == "valid":
			data_path = os.path.join(data_dir, "valid.txt")
		elif split == "test":
			data_path = os.path.join(data_dir, "test.txt")
		with open(data_path, encoding="utf8") as f:
			data = f.readlines()

		acc_add_nodes = 0
		for inst in data:
			inst = inst.strip()
			if inst:
				inst = inst.split('\t')
				if "conceptnet" in data_dir:
					if len(inst) == 3:
						rel, src, tgt = inst
						weight = 1.0
						label = 1
						src = src.lower()
						tgt = tgt.lower()
						if split != "train":
							_, new_added = self.add_example(src, tgt, rel, float(weight), int(label), train_network)
							acc_add_nodes += new_added
						else:
							self.add_example(src, tgt, rel, float(weight))
				elif "atomic" in data_dir:
					if len(inst) == 3:
						weight = 1.0
						label = 1
						src, rel, tgt = inst
						if split != "train":
							_, new_added = self.add_example(src, tgt, rel, float(weight), int(label), train_network)
							acc_add_nodes += new_added
				else:
					raise ValueError("Invalid option for dataset name.")


		logger.info('Number of OOV nodes in {}: {}'.format(split, acc_add_nodes))
		self.rel2id = self.graph.relations2id






def load_link_data(args, train_data_only=False, logger=None):
	if args.dataset == "atomic":
		data_dir = "./atomic"
	elif args.dataset == "conceptnet-100k":
		data_dir = "./conceptnet-100k"
	else:
		raise ValueError("Invalid option for dataset.")

	train_network = Reader_Data(args.dataset, logger)
	if not train_data_only:
		dev_network = Reader_Data(args.dataset, logger)
		test_network = Reader_Data(args.dataset, logger)

	train_network.read_data(data_dir= data_dir, split= "train", logger = logger)
	train_network.print_summary(logger= logger)
	node_list = train_network.graph.iter_nodes()

	node_degrees = [node.get_degree() for node in node_list]
	degree_counter = Counter(node_degrees)
	avg_degree = sum([k*v for k,v in degree_counter.items()]) / sum([v for k,v in degree_counter.items()])

	logger.info("Average Degree:{}".format(avg_degree))


	if not train_data_only:
		dev_network.read_data(data_dir=data_dir, split="valid",train_network=train_network, logger= logger)
		test_network.read_data(data_dir=data_dir, split="test", train_network=train_network, logger = logger)

	word_vocab = train_network.graph.nodes2id

	train_data, _ = utils.prepare_batch_dgl(word_vocab, train_network, train_network)
	if not train_data_only:
		dev_data, _ = utils.prepare_batch_dgl(word_vocab, dev_network, train_network)
		test_data, _ = utils.prepare_batch_dgl(word_vocab, test_network, train_network)
	id2node = {v:k for k,v in word_vocab.items()}

	logger.info("Total Nodes:{}".format(len(id2node)))
	if not train_data_only:
		return train_data, dev_data, test_data, train_network, id2node
	else:
		return train_data, train_network, id2node


def load_link_filter_data(args, degrees, original_data):
	filtered_data = []
	for item in original_data:
		if degrees[item[0]] == 0 or degrees[item[2]] == 0:
			filtered_data.append(item)
	return np.array(filtered_data)



from collections import defaultdict




class Node:

	def __init__(self, name, id, lang='en'):
		self.name = name
		self.id = id
		self.lang = lang
		self.neighbors = set([])

	def __str__(self):
		out = ("Node #%d : %s" % (self.id , self.name))
		return out

	def get_neighbors(self):

		return self.neighbors

	def get_degree(self):


		return len(self.neighbors)


class Relation:

	def __init__(self, name, id):
		self.name = name
		self.id = id

	def __str__(self):
		out = ("Relation #%d : %s" % (self.id , self.name))
		return out


class Edge:

	def __init__(self, node1, node2, relation, label, weight, uri):

		self.src = node1
		self.tgt = node2
		self.relation = relation
		self.label = label
		self.weight = weight
		self.uri = uri

	def __str__(self):
		out = ("Edge:(source, realtion, target) : (%s, %s, %s)" % (self.src.name, self.relation.name, self.tgt.name))
		return out


class Graph:

	def __init__(self, directed = True, logger = None):
		self.relations = defaultdict()
		self.relations2id = {}
		self.nodes = defaultdict()
		self.nodes2id = {}
		self.edges = defaultdict()
		self.edgeCount = 0
		self.directed = directed
		self.logger = logger


	def __str__(self):
		for node in self.nodes:
			self.logger.info("grah_class.py:{}".format(node))

	def add_edge(self, node1, node2, rel, label, weight, uri=None):


		new_edge = Edge(node1, node2, rel, label, weight, uri)

		if node2 in self.edges[node1]:
			self.edges[node1][node2].append(new_edge)
		else:
			self.edges[node1][node2] = [new_edge]

		node2.neighbors.add(node1)
		self.edgeCount += 1

		if (self.edgeCount+1) % 10000 == 0:
			self.logger.info("Number of edges :{}".format(self.edgeCount))

		return new_edge

	def add_node(self, name):

		new_node = Node(name, len(self.nodes))
		self.nodes[len(self.nodes)] = new_node
		self.nodes2id[new_node.name] = len(self.nodes) - 1
		self.edges[new_node] = {}
		return self.nodes2id[new_node.name]

	def add_relation(self, name):

		new_relation = Relation(name, len(self.relations))
		self.relations[len(self.relations)] = new_relation
		self.relations2id[new_relation.name] = len(self.relations) - 1
		return self.relations2id[new_relation.name]

	def find_node(self, name):

		if name in self.nodes2id:
			return self.nodes2id[name]
		else:
			return -1

	def find_relation(self, name):

		if name in self.relations2id:
			return self.relations2id[name]
		else:
			return -1

	def is_connected(self, node1, node2):

		if node1 in self.edges:
			if node2 in self.edges[node1][node2]:
				return True
		return False

	def node_exists(self, node):

		if node in self.nodes.values():
			return True
		return False

	def find_all_connections(self, relation):

		relevant_edges = []
		for edge in self.edges:
			if edge.relation == relation:
				relevant_edges.append(edge)

		return relevant_edges

	def iter_nodes(self):
		return list(self.nodes.values())

	def iter_relations(self):
		return list(self.relations.values())

	def iter_edges(self):
		for node in self.edges:
			for edge_list in self.edges[node].values():
				for edge in edge_list:
					yield edge


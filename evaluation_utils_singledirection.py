import time

import numpy as np
import torch

import json
import copy

import matplotlib.pyplot as plt

def entity_ranking_and_hits(args, model, test_triplets, e1_to_multi_e2, e2_to_multi_e1, network, write_results=False, debug=False, logger = None):

	s = test_triplets[:, 0]
	r = test_triplets[:, 1]
	o = test_triplets[:, 2]

	hits = []
	hits_e2 = []

	ranks = []
	ranks_e2 = []

	scores = []
	scores_e2 = []
	node_mrr = {}

	for i in range(50):
		hits_e2.append([])
		hits.append([])

	batch_size = 64

	if debug:
		end = min(5000, len(test_triplets))
	else:
		end = len(test_triplets)


	for i in range(0, end, batch_size):
		e1 = s[i : i+batch_size]
		e2 = o[i : i+batch_size]
		rel = r[i : i+batch_size]

		cur_batch_size = len(e1)




		e2_multi = [torch.LongTensor(e1_to_multi_e2[(e.cpu().item(), r.cpu().item())]) for e, r in zip(e1, rel)]

		with torch.no_grad():
			pred_e2 = model(e1=e1, rel=rel)


		pred_e2 = pred_e2.data.cpu()
		scores.append(pred_e2)

		e1, e2 = e1.data, e2.data

		for j in range(0,cur_batch_size):


			filter2 = e2_multi[j].long()

			target_e2 = pred_e2[j, e2[j].item()].item()


			pred_e2[j][filter2] = 0.0


			pred_e2[j][e1[j].item()] = 0.0

			pred_e2[j][e2[j]] = target_e2

			scores_e2.append(target_e2)

		max_values, argsort_e2 = torch.sort(pred_e2, 1, descending=True)

		for j in range(0,cur_batch_size):

			temp_rank_e2 = (argsort_e2[j] == e2[j]).nonzero().cpu().item()
			ranks.append(temp_rank_e2+1)
			ranks_e2.append(temp_rank_e2+1)


			node_e2 = network.graph.nodes[e2[j].cpu().item()]


			if node_e2 not in node_mrr:
				node_mrr[node_e2] = []

			node_mrr[node_e2].append(temp_rank_e2)

			for hits_level in range(0,50):

				if temp_rank_e2 <= hits_level:
					hits[hits_level].append(1.0)
					hits_e2[hits_level].append(1.0)
				else:
					hits[hits_level].append(0.0)
					hits_e2[hits_level].append(0.0)

	for k in [0, 2, 9, 19, 29, 39, 49]:
		logger.info('Hits@{}: {}'.format(k+1, np.mean(hits[k])))

	logger.info('Mean rank: {}'.format(np.mean(ranks)))
	logger.info('Mean reciprocal rank: {}'.format(np.mean(1.0 / np.array(ranks))))


	if write_results:
		write_topk_tuples(torch.cat(scores, dim=0).cpu().numpy(), test_triplets, network, ranks_e2, scores_e2)

	return np.mean(1.0 / np.array(ranks))




def write_topk_tuples(scores, input_prefs, network, ranks_e2, scores_e2, k=20):
	out_lines = []

	argsort = [np.argsort(-1 * np.array(score)) for score in np.array(scores)]
	for i,sorted_scores in enumerate(argsort):

		pref = input_prefs[i]
		e1 = pref[0].cpu().item()
		rel = pref[1].cpu().item()
		e2 = pref[2].cpu().item()

		cur_point = {}
		cur_point['gold_triple'] = {}
		cur_point['gold_triple']['e1'] = network.graph.nodes[e1].name
		cur_point['gold_triple']['relation'] = network.graph.relations[rel].name
		cur_point['gold_triple']['e2'] = network.graph.nodes[e2].name
		cur_point['gold_triple']['score_e2'] = scores_e2[i]
		cur_point['gold_triple']['rank_e2'] = ranks_e2[i]

		topk_indices = sorted_scores[:k]
		topk_tuples = [network.graph.nodes[elem] for elem in topk_indices]

		cur_point['candidates'] = []


		for j,node in enumerate(topk_tuples):
			tup= {}
			tup['e2'] = node.name
			tup['score'] = str([scores[i][topk_indices[j]]])
			cur_point['candidates'].append(tup)

		out_lines.append(cur_point)

	t = time.localtime()
	cur_time = time.strftime("%y_%m_%d_%H_%M_%S", t)
	with open('topk_candidates'+ str(cur_time) + '.jsonl', 'w') as f:
		for entry in out_lines:
			json.dump(entry, f)
			f.write('\n')



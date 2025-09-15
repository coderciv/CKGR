import os
import time
import tqdm




import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())





similar_entity = {}

with open('./conceptnet-100k/EA.txt.', 'r', encoding='utf-8') as txf:
	lines = txf.readlines()
	for line in lines:
		entitys = line.strip().split('\t')
		if entitys[0] not in similar_entity.keys():
			similar_entity[entitys[0]] = set([])
			similar_entity[entitys[0]].add(entitys[1])
		else:
			similar_entity[entitys[0]].add(entitys[1])


share_question1 = "A triple consists of a head entity, a relation, and a tail entity, expressed as (head entity, relation, tail entity). The current triple is "

share_tail_question1 = ". Candidate tail entities ("

share_head_question1 = ". Candidate head entities ("

share_question2 = " have been calculated. "

share_tail_question2 = "Please select a suitable replacement from these candidate tail entities and assign a score between 0 and 1 based on the rationality of the replaced triple. " \
                       "If a candidate tail entity does not conform to common sense, it will be assigned 0 points, and no additional construction of new triples is allowed." \
                       "Only quadruplets in the format of (head entity, relation, tail entity, score) are returned. The number of quadruplets must be equal to the number of candidate tail entities. " \
                       "Each quadruplet is separated by a line break and does not require additional explanation."




share_head_question2 = "Please select a suitable replacement from these candidate head entities and assign a score between 0 and 1 based on the rationality of the replaced triple. " \
                       "If a candidate head entity does not conform to common sense, it will be assigned 0 points, and no additional construction of new triples is allowed." \
                       "Only quadruplets in the format of (head entity, relation, tail entity, score) are returned. The number of quadruplets must be equal to the number of candidate head entities. " \
                       "Each quadruplet is separated by a line break and does not require additional explanation."




openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo"):
	messages = [{"role": "user", "content": prompt}]
	response = openai.chat.completions.create(
	model=model,
	messages=messages,
	return response.choices[0].message.content




before_index = 0
after_index = 0
augmentdata = []


with open('./conceptnet-100k/train.txt.', 'r', encoding='utf-8') as txf:
	lines = txf.readlines()
	for line in tqdm.tqdm(lines, ncols=85):
		element = line.strip().split('\t')
		original_triple = "("+str(element[1])+","+str(element[0])+","+str(element[2])+",2)"
		augmentdata.append(original_triple)

		if element[1] in similar_entity.keys():
			candidate_head_entity = "("
			for entity in similar_entity[element[1]]:
				candidate_head_entity = candidate_head_entity + str(entity) + ","
			candidate_head_entity = candidate_head_entity[:-1] + ")"

			question = share_question1 + original_triple + share_head_question1 + candidate_head_entity + share_question2 +share_head_question2

			result = get_completion(question).strip().split('\n')

			for candidate_triple in result:
				augmentdata.append(candidate_triple)

		if element[2] in similar_entity.keys():
			candidate_tail_entity = "("
			for entity in similar_entity[element[2]]:
				candidate_tail_entity = candidate_tail_entity + str(entity) + ","
			candidate_tail_entity = candidate_tail_entity[:-1] + ")"

			question = share_question1 + original_triple + share_tail_question1 + candidate_tail_entity + share_question2 +share_tail_question2

			result = get_completion(question).strip().split('\n')

			for candidate_triple in result:
				augmentdata.append(candidate_triple)









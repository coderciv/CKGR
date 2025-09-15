import os
import sys
import random
from copy import deepcopy

import numpy as np
import torch
import time



import parse_args
import utils
import data_loader
import evaluation_utils

from model import LinkPredictor





def main(args,logger):


    utils.set_seeds(args.seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda and not args.no_cuda:
        args.device = torch.device(args.cuda_id)
    else:
        args.device = torch.device("cpu")




    train_data, valid_data, test_data, train_network, id2node = data_loader.load_link_data(args, logger = logger)


    num_nodes = len(train_network.graph.nodes)
    num_rels = len(train_network.graph.relations)
    args.num_nodes = num_nodes
    args.num_rels = num_rels



    if args.dataset == "atomic":
        valid_data = valid_data[:10000]
        test_data = test_data[:10000]

    logger.info("Train Triple :{}".format(train_data.shape[0]))
    logger.info("Val Triple :{}".format(valid_data.shape[0]))
    logger.info("Test Triple :{}".format(test_data.shape[0]))


    _, degrees, _, _ = utils.get_adj_and_degree(num_nodes, num_rels, train_data)
    inductive_index = (np.where(degrees == 0)[0]).tolist()

    filtered_valid = data_loader.load_link_filter_data(args, degrees, valid_data)
    filtered_test = data_loader.load_link_filter_data(args, degrees, test_data)

    logger.info('NA Val Triplet #: %d', filtered_valid.shape[0])
    logger.info('NA Test Triplet #: %d', filtered_test.shape[0])




    all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()


    all_e1_to_multi_e2, all_e2_to_multi_e1 = utils.create_entity_dicts(args, all_tuples, num_rels)

    train_e1_to_multi_e2, train_e2_to_multi_e1 = utils.create_entity_dicts(args, train_data.tolist(), num_rels)

    filtered_valid = torch.LongTensor(filtered_valid)
    filtered_test = torch.LongTensor(filtered_test)
    filtered_valid = filtered_valid.to(args.device)
    filtered_test = filtered_test.to(args.device)

    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    valid_data = valid_data.to(args.device)
    test_data = test_data.to(args.device)




    if args.bert_feat_path != 'None' and args.fasttext_feat_path != 'None':
        bert_feature = utils.load_pre_computed_feat(args.bert_feat_path, args.bert_feat_dim, id2node, logger)
        fasttext_feature = utils.load_pre_computed_feat(args.fasttext_feat_path, args.fasttext_feat_dim, id2node, logger)
        fusion_feature = torch.cat((bert_feature, fasttext_feature),dim=1)
        logger.info("Loading Pre-computed BERT and fasttext Embedding")
    elif args.bert_feat_path != 'None':
        bert_feature = utils.load_pre_computed_feat(args.bert_feat_path, args.bert_feat_dim, id2node, logger)
        logger.info("Loading Pre-computed BERT Embedding")
    elif args.fasttext_feat_path != 'None':
        fasttext_feature = utils.load_pre_computed_feat(args.fasttext_feat_path, args.fasttext_feat_dim, id2node, logger)
        logger.info("Loading Pre-computed fasttext Embedding")
    else:
        logger.info("No node feature provided. Use random initialization")
    logger.info('')



    fix_edge_src = []
    fix_edge_tgt = []
    fix_edge_type = []

    args.num_edge_types = 0

    if args.fix_triplet_graph:
        tri_edge_src, tri_edge_tgt, tri_edge_type = utils.create_triplet_graph(args, train_data.tolist(), logger)
        tri_graph = (tri_edge_src, tri_edge_tgt, tri_edge_type)
        logger.info('Number of triplet edges: {}'.format(len(tri_edge_src)))
        if args.inverse_relation:
            logger.info('Number of triplet edges types with inverse relation: {}'.format(args.num_rels * 2))
        else:
            logger.info('Number of triplet edges types without inverse relation: {}'.format(args.num_rels))
        logger.info('')

        fix_edge_src.extend(tri_graph[0])
        fix_edge_tgt.extend(tri_graph[1])
        fix_edge_type.extend(tri_graph[2])

    if args.fix_triplet_graph:
        args.num_edge_types = args.num_edge_types + args.num_rels

    if args.inverse_relation:
        logger.info('Add inverse edge type for semantic similarity graph')
        args.num_edge_types = args.num_edge_types + args.num_rels

    if args.dynamic_sim_graph:
        logger.info('Add sim edge type for semantic similarity graph')
        args.num_edge_types = args.num_edge_types + 1

    logger.info('Number of relation types for R-GCN model：{}'.format(args.num_edge_types))

    fixed_graph = (fix_edge_src, fix_edge_tgt, fix_edge_type)
    logger.info('Total number of fixed edges: {}'.format(len(fix_edge_src)))
    logger.info('')





    epoch = 0

    model = LinkPredictor(args)
    if args.load_model:
        model_state_file = args.load_model
        checkpoint = torch.load(model_state_file)
        model.load_state_dict(checkpoint['state_dict'])



    logger.info("model infomation:{}".format(model))

    if args.bert_feat_path != 'None' and args.fasttext_feat_path != 'None':
        logger.info("Initialize with concatenated BERT and fastText Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': fusion_feature})
    elif args.bert_feat_path != 'None':
        logger.info("Initialize with Pre-computed BERT Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': bert_feature})
    elif args.fasttext_feat_path != 'None':
        logger.info("Initialize with Pre-computed fasttext Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': fasttext_feature})
    else:
        logger.info("No node feature provided. Use uniform initialization")

    model.to(args.device)



    t = time.localtime()
    cur_time = time.strftime("%y_%m_%d_%H_%M_%S",t)



    args.model_state_file = os.path.join(args.output_dir, cur_time + ".pt")
    logger.info("Model Sava Path: {}".format(args.model_state_file))


    if os.path.isfile(args.model_state_file):
        logger.info(args.model_state_file)
        if not args.overwrite:
            logger.info('Model already exists. Use Overwrite')
            sys.exit(0)


    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=args.regularization)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)
    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    entity_forward_time = []
    entity_backward_time = []



    logger.info("Starting training...")

    entity_best_mrr = 0
    entity_f_best_mrr = 0
    patient_times = 0
    batch_size = args.decoder_batch_size


    current_graph = fixed_graph

    fixed_train_data = train_data[:]
    logger.info("Number of fixed training data: {}".format(fixed_train_data.shape[0]))

    while True:
        epoch += 1



        if args.dataset == 'atomic' and epoch%args.evaluate_every == 1 and epoch != 1:

            degrees_copy = deepcopy(degrees)
            sim_edge_src, sim_edge_tgt, sim_edge_type = utils.dynamic_graph_gen(args,
                                                                                model.entity_embedding.detach().cpu().numpy(),
                                                                                n_ontology=args.n_ontology,
                                                                                inductive_index=inductive_index,
                                                                                degrees=degrees_copy,
                                                                                logger= logger)
            sim_graph = (sim_edge_src, sim_edge_tgt, sim_edge_type)

            cur_edge_src = []
            cur_edge_tgt = []
            cur_edge_type = []

            cur_edge_src.extend(fixed_graph[0])
            cur_edge_tgt.extend(fixed_graph[1])
            cur_edge_type.extend(fixed_graph[2])

            cur_edge_src.extend(sim_graph[0])
            cur_edge_tgt.extend(sim_graph[1])
            cur_edge_type.extend(sim_graph[2])

            current_graph = (cur_edge_src, cur_edge_tgt, cur_edge_type)

            logger.info('Number of edges in augmented graph with sim and syn_triplets: {}'.format(len(current_graph[0])))

        if args.dataset[:10] == 'conceptnet' and args.dynamic_sim_graph and epoch>args.start_dynamic_graph and epoch%args.dynamic_graph_ee_epochs ==1:
            logger.info('')
            logger.info('********************* Update Graph************************')

            logger.info("Update dynamic similarity graph")



            g_whole, node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, current_graph,logger, test_graph_bool=True)
            node_id_copy = np.copy(node_id)
            if model.entity_embedding != None:
                del model.entity_embedding
                model.entity_embedding = None

            model.eval()
            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                model.cpu()
                g_whole = g_whole.cpu()
                g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
                g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
                g_whole.edata['type'] = g_whole.edata['type'].cpu()

            torch.cuda.empty_cache()
            model.update_whole_embedding_matrix(g_whole,node_id_copy)

            degrees_copy = deepcopy(degrees)
            sim_edge_src, sim_edge_tgt, sim_edge_type = utils.dynamic_graph_gen(
                args,
                model.entity_embedding.detach().cpu().numpy(),
                n_ontology=args.n_ontology,
                inductive_index=inductive_index,
                degrees=degrees_copy,
                logger = logger
            )

            sim_graph = (sim_edge_src, sim_edge_tgt, sim_edge_type)


            cur_edge_src = []
            cur_edge_tgt = []
            cur_edge_type = []


            cur_edge_src.extend(fixed_graph[0])
            cur_edge_tgt.extend(fixed_graph[1])
            cur_edge_type.extend(fixed_graph[2])

            cur_edge_src.extend(sim_graph[0])
            cur_edge_tgt.extend(sim_graph[1])
            cur_edge_type.extend(sim_graph[2])

            current_graph = (cur_edge_src, cur_edge_tgt, cur_edge_type)
            logger.info("Number of edges in augmented graph with sim and syn_triplets: {}".format(len(current_graph[0])))

            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                logger.info('Transfer model back to: {}'.format(args.device))
                model.to(args.device)

            logger.info("**************** Finish Updating Graph *******************************")
            logger.info('')



        model.train()

        logger.info("**********************************************************************")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_all = sum(p.numel() for p in model.parameters())


        total_params_in_million = total_params / 1e6
        total_params_all_in_million = total_params_all / 1e6

        logger.info("Total number of trainable parameters: {}M".format(total_params_in_million))
        logger.info("Total number of parameters (including non-trainable): {}M".format(total_params_all_in_million))
        logger.info("**********************************************************************")




        g, node_id ,node_norm = utils.sample_sub_graph(args, args.graph_batch_size, current_graph, logger, test_graph_bool=False)
        node_id_copy = np.copy(node_id)
        node_dict = {v :k for k,v in dict(enumerate(node_id_copy)).items()}



        cur_train_data = train_data[:]

        rel = cur_train_data[:, 1]

        if args.inverse_relation:
            src, dst = np.concatenate((cur_train_data[:,0], cur_train_data[:,2])),\
                   np.concatenate((cur_train_data[:,2], cur_train_data[:,0]))
            rel = np.concatenate((rel,rel+num_rels))
        else:
            src, dst = np.concatenate((cur_train_data[:, 0], cur_train_data[:, 2]))

        cur_train_data = np.stack((src, rel, dst)).transpose()


        graph_e2_keys = {}
        for triplet in cur_train_data:
            head = triplet[0]
            temp_rel = triplet[1]
            tail = triplet[2]

            if head in node_id_copy and tail in node_id_copy:
                subgraph_src_idx = node_dict[head]
                subgraph_tgt_idx = node_dict[tail]
                if (subgraph_src_idx, temp_rel) not in graph_e2_keys:
                    graph_e2_keys[(subgraph_src_idx, temp_rel)] = [subgraph_tgt_idx]
                else:
                    graph_e2_keys[(subgraph_src_idx, temp_rel)].append(subgraph_tgt_idx)




        entity_key_list = list(graph_e2_keys.keys())
        random.shuffle(entity_key_list)


        entity_cum_loss = 0.0
        for j in range(0, len(entity_key_list), batch_size):

            optimizer.zero_grad()

            batch = entity_key_list[j: j+batch_size]
            if len(batch) == 1:
                continue

            e1 = torch.LongTensor([elem[0] for elem in batch])
            rel = torch.LongTensor([elem[1] for elem in batch])


            predict_e2 = [graph_e2_keys[(elem[0], elem[1])] for elem in batch]
            batch_len = len(batch)


            entity_target = torch.zeros((batch_len, node_id_copy.shape[0]))

            e1 = e1.to(args.device)
            rel = rel.to(args.device)
            entity_target = entity_target.to(args.device)


            for index, inst in enumerate(predict_e2):
                entity_target[index, inst] = 1.0



            entity_target = ((1.0 - args.label_smoothing_epsilon)*entity_target) + (1.0/entity_target.size(1))

            t0 = time.time()

            if j % args.clean_update  == 0 :
                model.update_whole_embedding_matrix(g, node_id_copy)

            sample_normalization = None


            entity_loss = model(e1= e1, rel=rel, entity_target= entity_target, sample_normalization= sample_normalization)

            entity_cum_loss += entity_loss.cpu().item()

            t1 = time.time()
            entity_loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_norm)
            optimizer.step()

            t2 = time.time()
            entity_forward_time.append(t1 - t0)
            entity_backward_time.append(t2 - t1)

        t = time.localtime()
        current_time = time.strftime("%D - %H:%M:%S", t)
        logger.info("Entity Predict Information {} | Epoch {:d} | Loss {:.4f} | Best MRR {:.4f} | Best fMRR {:.4f}| Forward {:.4f}s | Backward {:.4f}s | lr {}".
            format(current_time, epoch, entity_cum_loss, entity_best_mrr, entity_f_best_mrr, entity_forward_time[-1], entity_backward_time[-1], optimizer.param_groups[0]['lr']))




        if epoch % args.evaluate_every == 0 :
            model.eval()
            logger.info("\n")
            logger.info("******************start eval*********************")

            eval_edge_src = []
            eval_edge_tgt = []
            eval_edge_type = []

            eval_edge_src.extend(fixed_graph[0])
            eval_edge_tgt.extend(fixed_graph[1])
            eval_edge_type.extend(fixed_graph[2])

            logger.info("Generate graph for evaluation")

            logger.info("First pass for all entity embedding")

            g_whole ,node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, current_graph,  logger= logger, test_graph_bool=True)

            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                logger.info('perform evaluation on cpu')

                model.cpu()
                g_whole = g_whole.cpu()
                g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
                g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
                g_whole.edata['type'] = g_whole.edata['type'].cpu()
                valid_data = valid_data.cpu()
                test_data = test_data.cpu()
                filtered_valid = filtered_valid.cpu()
                filtered_test = filtered_test.cpu()

            node_id_copy = np.copy(node_id)

            if model.entity_embedding != None:
                del model.entity_embedding
                model.entity_embedding = None

            torch.cuda.empty_cache()
            model.update_whole_embedding_matrix(g_whole, node_id_copy)

            if args.dynamic_sim_graph:
                logger.info('Create similarity edges')
                degrees_copy = deepcopy(degrees)
                sim_edge_src, sim_edge_tgt, sim_edge_type=utils.dynamic_graph_gen(
                    args,
                    model.entity_embedding.detach().cpu().numpy(),
                    n_ontology=args.n_ontology,
                    inductive_index=[],
                    degrees=degrees_copy,
                    logger = logger)

                sim_graph = (sim_edge_src, sim_edge_tgt, sim_edge_type)

                eval_edge_src.extend(sim_graph[0])
                eval_edge_tgt.extend(sim_graph[1])
                eval_edge_type.extend(sim_graph[2])

            eval_graph =  (eval_edge_src, eval_edge_tgt, eval_edge_type)
            logger.info('Number of edges evalution graph:{}'.format(len(eval_graph[0])))


            g_whole, node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, eval_graph, logger=logger, test_graph_bool=True)




            logger.info('Update whole entity embedding from generated graph')
            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                g_whole = g_whole.cpu()
                g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
                g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
                g_whole.edata['type'] = g_whole.edata['type'].cpu()

            node_id_copy = np.copy(node_id)
            if model.entity_embedding != None:
                del model.entity_embedding
                model.entity_embedding = None
            torch.cuda.empty_cache()


            model.update_whole_embedding_matrix(g_whole, node_id_copy)

            logger.info('')
            logger.info('========DEV==========')
            entity_mrr_dev = evaluation_utils.entity_ranking_and_hits(args, model, valid_data, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)
            logger.info("================TEST================")
            entity_mrr_test = evaluation_utils.entity_ranking_and_hits(args, model, test_data, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)
            logger.info("=========== Filtered DEV============")
            entity_f_mrr_dev = evaluation_utils.entity_ranking_and_hits(args, model, filtered_valid, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)
            logger.info("================Filtered TEST================")
            entity_f_mrr_test = evaluation_utils.entity_ranking_and_hits(args, model, filtered_test, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)


            scheduler.step(entity_mrr_dev)


            if entity_mrr_dev < entity_best_mrr:
                if epoch>=args.n_epochs:
                    patient_times += 1
                    if patient_times >= args.patient:
                        logger.info('Early stopping....')
                        break
            else:
                patient_times = 0
                entity_best_mrr = entity_mrr_dev
                logger.info('[Saving best model so far]')
                torch.save({'state_dict':model.state_dict(), 'epoch':epoch, 'eval_graph':eval_graph}, args.model_state_file)

            if entity_f_mrr_dev > entity_f_best_mrr:
                patient_times = 0
                entity_f_best_mrr = entity_f_mrr_dev
                logger.info('[Saving best model so far]')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'eval_graph': eval_graph}, args.model_state_file + 'pt')

            logger.info('Current patient: {}'.format(patient_times))



            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                logger.info('Transfer model back to:{}'.format(args.device))
                model.to(args.device)
                g_whole = g_whole.to(args.device)
                g_whole.ndata['id'] = g_whole.ndata['id'].to(args.device)
                g_whole.ndata['norm'] = g_whole.ndata['norm'].to(args.device)
                g_whole.edata['type'] = g_whole.edata['type'].to(args.device)

                valid_data = valid_data.to(args.device)
                test_data = test_data.to(args.device)
                filtered_valid = filtered_valid.to(args.device)
                filtered_test = filtered_test.to(args.device)

            logger.info('****************end eval*****************')
            logger.info(' ')
            torch.cuda.empty_cache()


    logger.info('Training done')
    logger.info('Mean all forward time:{:4f}s'.format(np.mean(entity_forward_time)))
    logger.info('Mean all backward time:{:4f}s'.format(np.mean(entity_backward_time)))

    logger.info('Start test (1)')

    checkpoint = torch.load(args.model_state_file)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    eval_graph = checkpoint['eval_graph']

    logger.info("using best epoch: {}".format(checkpoint['epoch']))

    g_whole,node_id,node_norm = utils.sample_sub_graph(args, 99999999999999, eval_graph, logger=logger, test_graph_bool=True)
    if args.dataset == 'atomic' or args.dataset[:10]=='conceptnet':
        logger.info('perform evaluation on cpu')
        model.cpu()
        g_whole = g_whole.cpu()
        g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
        g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
        g_whole.edata['type'] = g_whole.edata['type'].cpu()

        valid_data = valid_data.cpu()
        test_data = test_data.cpu()
        filtered_valid = filtered_valid.cpu()
        filtered_test = filtered_test.cpu()

    if model.entity_embedding != None:
        del model.entity_embedding
        model.entity_embedding = None
        torch.cuda.empty_cache()
    node_id_copy = np.copy(node_id)
    model.update_whole_embedding_matrix(g_whole, node_id_copy)

    logger.info('==============DEV===============')
    evaluation_utils.entity_ranking_and_hits(args, model, valid_data, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)
    logger.info('==============TEST===============')
    evaluation_utils.entity_ranking_and_hits(args, model, test_data, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)






    logger.info('Start test (2)')

    checkpoint = torch.load(args.model_state_file + 'pt')
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    eval_graph = checkpoint['eval_graph']

    logger.info("using best epoch: {}".format(checkpoint['epoch']))

    g_whole, node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, eval_graph, logger=logger, test_graph_bool=True)
    if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
        logger.info('perform evaluation on cpu')
        model.cpu()
        g_whole = g_whole.cpu()
        g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
        g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
        g_whole.edata['type'] = g_whole.edata['type'].cpu()

        valid_data = valid_data.cpu()
        test_data = test_data.cpu()
        filtered_valid = filtered_valid.cpu()
        filtered_test = filtered_test.cpu()

    if model.entity_embedding != None:
        del model.entity_embedding
        model.entity_embedding = None
        torch.cuda.empty_cache()
    node_id_copy = np.copy(node_id)
    model.update_whole_embedding_matrix(g_whole, node_id_copy)

    logger.info('==============Filtered DEV===============')
    evaluation_utils.entity_ranking_and_hits(args, model, filtered_valid, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)
    logger.info('==============Filtered TEST===============')
    evaluation_utils.entity_ranking_and_hits(args, model, filtered_test, all_e1_to_multi_e2, all_e2_to_multi_e1, train_network, logger=logger)


if __name__ == '__main__':


    args,logger = parse_args.parse_args()

    try:
        main(args,logger)
    except KeyboardInterrupt:
        logger('Interrupted')
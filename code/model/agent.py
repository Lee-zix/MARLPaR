import numpy as np
import tensorflow as tf


class Agent(object):

    def __init__(self, params):

        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        with tf.variable_scope("action_lookup_table"):
            self.action_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.variable_scope("entity_lookup_table"):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        with tf.variable_scope("policy_step"):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.contrib.rnn.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
            self.policy_step = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state):
        with tf.variable_scope("MLP_for_policy_rel"):
            hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.layers.dense(hidden,2 * self.embedding_size, activation=tf.nn.relu)
        return output

    def policy_MLP_ENT(self, state):
        with tf.variable_scope("MLP_for_policy_ent"):
            hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.layers.dense(hidden,2 * self.embedding_size, activation=tf.nn.relu)
        return output

    def action_encoder(self, next_relations, next_entities):
        with tf.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def relation_encoder(self,next_relations):
        with tf.variable_scope("lookup_table_rel_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)  #[B,D]
        return relation_embedding

    def entity_encoder(self,next_entities):
        with tf.variable_scope("lookup_table_ent_edge_encoder"):
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)  #[B,D]
        return entity_embedding

    def step(self, next_possible_relations_weight, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
             label_action, range_arr, first_step_of_test):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        # 1. one step of rnn
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # output: [B, 4D]

        # Get state vector
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        if self.use_entity_embeddings:
            state_rel = tf.concat([output, prev_entity], axis=-1)
        else:
            state_rel = output

        candidate_relation_embeddings = self.relation_encoder(next_relations)
        state_query_concat = tf.concat([state_rel, query_embedding], axis=-1)

        # Get state vector for entity
        state_ent = tf.concat([output, prev_entity], axis=-1)
        state_ent_query_concat = tf.concat([state_ent, query_embedding], axis=-1)

        # MLP for policy entity#
        output_ent = self.policy_MLP_ENT(state_ent_query_concat)
        output_expanded_ent = tf.expand_dims(output_ent, axis=1)  # [B, 1, 2D]


        # MLP for policy relation#
        output_rel = self.policy_MLP(state_query_concat)
        output_expanded = tf.expand_dims(output_rel, axis=1)  # [B, 1, 2D]


        prelim_scores = tf.reduce_sum(tf.multiply(candidate_relation_embeddings, output_expanded), axis=2)


        #The following is the case where the model randomly satisfies the probability
        # distribution of r, and an entity is randomly selected from it
        prelim_scores = tf.multiply(prelim_scores, next_possible_relations_weight)


        #The following is to let the model in the case of a given r, policy an entity,
        # 1 or choose  the relationship, and count the score

        # Masking PAD actions

        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]

        # 4 sample action
        action = tf.to_int32(tf.multinomial(logits=scores, num_samples=1))  # [B, 1]
        # loss
        # 5a.
        label_action =  tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]

        # 6. Map back to true id
        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        #action_idx
        temp = tf.ones([1,next_relations.shape[1]],dtype=tf.int32)
        chosen_relation_expand = tf.multiply(action, temp)
        re = tf.bitwise.bitwise_xor(next_relations, chosen_relation_expand) #[[0, 3, 3, 3],[7, 0, 6, 5],[0, 0, 3, 3],[0, 3, 3, 3]]
        re1 = tf.equal(re, 0)
        temp = tf.ones_like(next_entities, dtype=tf.int32)
        next_filter_entities = tf.where(re1,next_entities,temp)
        candidate_entity_embeddings = self.entity_encoder(next_filter_entities)
        prelim_ent_scores = tf.reduce_sum(tf.multiply(candidate_entity_embeddings, output_expanded_ent), axis=2)

        comparison_tensor = tf.ones_like(next_filter_entities, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_filter_entities, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_ent_scores) * -99999.0  # the base matrix to choose from if dummy relation
        ent_scores = tf.where(mask, dummy_scores, prelim_ent_scores)  # [B, MAX_NUM_ACTIONS]


        # 4 sample action
        ent_action = tf.to_int32(tf.multinomial(logits=ent_scores, num_samples=1))  # [B, 1]
        # loss
        ent_label_action =  tf.squeeze(ent_action, axis=1)
        ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ent_scores, labels=ent_label_action)  # [B,]

        # 6. Map back to true id
        action_idx = tf.squeeze(ent_action)
        chosen_triple = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, ent_loss, new_state, tf.nn.log_softmax(scores),  tf.nn.log_softmax(ent_scores), action_idx, chosen_triple

    def __call__(self,candidate_relation_weight_sequence ,candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 path_label, query_relation, range_arr, first_step_of_test, T=3, entity_sequence=0):

        self.baseline_inputs = []
        # get the query vector
        query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)  # [B, 2D]
        state = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        prev_relation = self.dummy_start_label

        all_rel_loss = []  # list of loss tensors each [B,]
        all_ent_loss = []  # list of actions each [B,]
        all_rel_logits = []
        all_ent_logits = []

        action_idx = []

        with tf.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]  # [B, MAX_NUM_ACTIONS, MAX_EDGE_LENGTH] --t-->  [B, MAX_NUM_ACTIONS]
                next_possible_relations_weight = candidate_relation_weight_sequence[t]
                next_possible_entities = candidate_entity_sequence[t]
                current_entities_t = current_entities[t]                  # [B, MAX_EDGE_LENGTH] --t-->  [B]

                path_label_t = path_label[t]  # [B]
                #loss, ent_loss, new_state, tf.nn.log_softmax(scores), tf.nn.log_softmax(
                 #   ent_scores), action_idx, chosen_triple
                loss, ent_loss, state, logits, ent_logits, idx, chosen_relation = self.step(next_possible_relations_weight,
                                                                              next_possible_relations,
                                                                              next_possible_entities,
                                                                              state, prev_relation, query_embedding,
                                                                              current_entities_t,
                                                                              label_action=path_label_t,
                                                                              range_arr=range_arr,
                                                                              first_step_of_test=first_step_of_test)

                all_rel_loss.append(loss)
                all_ent_loss.append(ent_loss)
                all_rel_logits.append(logits)
                all_ent_logits.append(ent_logits)
                action_idx.append(idx)
                prev_relation = chosen_relation

            # [(B, T), 4D]

        return all_rel_loss, all_ent_loss, all_rel_logits, all_ent_logits, action_idx

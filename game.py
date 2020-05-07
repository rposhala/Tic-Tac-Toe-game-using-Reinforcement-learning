import random
from collections import deque

import numpy as np
import tensorflow.compat.v1 as tf

from gui import GUI
import matplotlib.pyplot as plt

MODEL_FOLDER_NAME = "model"
# MODEL_FOLDER_NAME = "/gdrive/My Drive/model"
LOGS_SAVE_LOCATION = "/gdrive/My Drive/logs/value_network"
SAVED_MODELS_LOCATION = "/gdrive/My Drive/model/saved_network"

REPLAY_QUEUE_SIZE = 10000

TARGET_UPDATE_RATE, REGULARIZATION_CONSTANT = 0.01, 0.01

SUMMARY_PRINT_RATE = 100

LEARNING_RATE = 0.2
DISCOUNT_RATE = 0.9
ANNEAL_EPSILON = 10000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 0.7
BATCH_UPDATE_SIZE = 32
EMPTY_BOARD_SPACE = -1
TRAINING_ITERATIONS = 20000

REWARD_LOSE = -1
REWARD_DRAW = 0.5
REWARD_WIN = 1
BOARD_DIMENSIONS = 9
TOTAL_ACTIONS = 9

WINNING_COMBOS = ([6, 7, 8], [3, 4, 5], [0, 1, 2], [0, 3, 6], [1, 4, 7], [2, 5, 8],
                  [0, 4, 8], [2, 4, 6])


class MemoryBuffer(object):

    def __init__(self, queue_length):
        self.buffer = deque()
        self.size = 0
        self.queue_length = queue_length

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def push(self, exp):
        if self.size < self.queue_length:
            self.buffer.append(exp)
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp)


class DeepAgent(object):

    def __init__(self, player, session,
                 optimizer, store_replay_every=5,
                 max_gradient=5, summary_writer=None):
        self.player = player

        self.session = session
        self.optimizer = optimizer
        self.summary_writer = summary_writer

        self.memory_buffer = MemoryBuffer(queue_length=REPLAY_QUEUE_SIZE)

        self.batch_size = BATCH_UPDATE_SIZE
        self.exploration = INITIAL_EPSILON
        self.init_exp = INITIAL_EPSILON
        self.final_exp = FINAL_EPSILON
        self.anneal_steps = ANNEAL_EPSILON
        self.discount_factor = DISCOUNT_RATE
        self.target_update_rate = TARGET_UPDATE_RATE

        self.max_gradient = max_gradient
        self.reg_param = REGULARIZATION_CONSTANT

        self.store_replay_every = store_replay_every
        self.store_experience_cnt = 0
        self.train_iteration = 0

        with tf.name_scope("predict_actions"):
            self.states = tf.placeholder(tf.float32, (None, BOARD_DIMENSIONS), name="states")
            with tf.variable_scope("q_network"):
                self.q_outputs = create_network(self.states, self.player)
            self.action_scores = tf.identity(self.q_outputs, name="action_scores")
            self.predicted_actions = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

        # Calculate rewards using next state: r(s_t,a_t) + argmax_a Q(s_{t+1}, a)
        with tf.name_scope("estimate_future_rewards"):
            self.next_states = tf.placeholder(tf.float32, (None, BOARD_DIMENSIONS), name="next_states")
            self.next_state_mask = tf.placeholder(tf.float32, (None,), name="next_state_masks")

        with tf.variable_scope("target_network"):
            self.target_outputs = create_network(self.next_states, self.player)
            self.next_action_scores = tf.stop_gradient(self.target_outputs)
            self.target_values = tf.reduce_max(self.next_action_scores,
                                               reduction_indices=[1, ]) * self.next_state_mask

            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
            self.future_rewards = self.rewards + self.discount_factor * self.target_values

        with tf.name_scope("compute_temporal_differences"):
            self.action_mask = tf.placeholder(tf.float32, (None, TOTAL_ACTIONS), name="action_mask")
            self.masked_action_scores = tf.reduce_sum(self.action_scores * self.action_mask,
                                                      reduction_indices=[1, ])
            # Defining the mean-squared loss function
            self.temp_diff = self.masked_action_scores - self.future_rewards
            self.td_loss = tf.reduce_mean(tf.square(self.temp_diff))
            # regularization loss
            learning_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                           scope="q_network")
            self.reg_loss = self.reg_param * tf.reduce_sum(
                [tf.reduce_sum(tf.square(x)) for x in learning_network_variables])
            self.loss = self.td_loss + self.reg_loss
            gradients = self.optimizer.compute_gradients(self.loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)
            # add histograms for gradients.
            for grad, var in gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)
            self.train_op = self.optimizer.apply_gradients(gradients)

        # update target network with Q-Learning network
        with tf.name_scope("update_target_network"):
            self.target_network_update = []
            # slowly update target network parameters with Q network parameters
            learning_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            target_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
            for v_source, v_target in zip(learning_network_variables, target_network_variables):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        self.summarize = tf.summary.merge_all()
        self.no_op = tf.no_op()

        tf.summary.histogram("action_scores", self.action_scores)
        tf.summary.histogram("next_action_scores", self.next_action_scores)

        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        self.session.run(tf.assert_variables_initialized())

        if self.summary_writer is not None:
            self.summary_writer.add_graph(self.session.graph)
            self.summary_every = SUMMARY_PRINT_RATE

    def greedy_action(self, states, explore=True):
        available_actions = list()
        for i in range(0, len(states[0])):
            s = states[0][i]
            if s == EMPTY_BOARD_SPACE:
                available_actions.append(i)
        if explore and self.exploration > np.random.random():
            return np.random.choice(available_actions)
        else:
            scores = self.session.run(self.action_scores, {self.states: states})[0]
            q = list()
            for a in available_actions:
                q.append(scores[a])
            idx = np.argmax(q)
            greedy_action = available_actions[int(idx)]
            return greedy_action

    def anneal_explorations(self):
        ratio = max((self.anneal_steps - self.train_iteration) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def update_model(self, props, done):
        if self.store_experience_cnt % self.store_replay_every == 0 or done:
            self.memory_buffer.push(props + (done,))
        self.store_experience_cnt += 1

        if self.memory_buffer.size < self.batch_size:
            return

        batch = self.memory_buffer.sample_batch(self.batch_size)
        states = np.zeros((self.batch_size, BOARD_DIMENSIONS))
        rewards = np.zeros((self.batch_size,))
        action_mask = np.zeros((self.batch_size, TOTAL_ACTIONS))
        next_states = np.zeros((self.batch_size, BOARD_DIMENSIONS))
        next_state_mask = np.zeros((self.batch_size,))

        for k, (s0, a, r, s1, done) in enumerate(batch):
            states[k] = s0
            rewards[k] = r
            action_mask[k][a] = 1
            if not done:
                next_states[k] = s1
                next_state_mask[k] = 1

        calculate_summaries = self.train_iteration % SUMMARY_PRINT_RATE == 0 and self.summary_writer is not None

        cost, _, summary_str = self.session.run([
            self.loss,
            self.train_op,
            self.summarize if calculate_summaries else self.no_op
        ], {
            self.states: states,
            self.next_states: next_states,
            self.next_state_mask: next_state_mask,
            self.action_mask: action_mask,
            self.rewards: rewards
        })

        self.session.run(self.target_network_update)

        if calculate_summaries:
            self.summary_writer.add_summary(summary_str, self.train_iteration)

        self.anneal_explorations()
        self.train_iteration += 1


class Environment:
    def __init__(self):
        self.board = [-1.0] * 9
        self.corners = [0, 2, 6, 8]
        self.sides = [1, 3, 5, 7]
        self.middle = 4
        self.p1_marker, self.p2_marker = self.get_marker()

    @staticmethod
    def get_marker():
        return 1.0, 0.0

    def reset(self):
        self.board = [-1.0] * 9
        return self.board

    def step(self, action, marker):
        over = False
        reward = 0

        self.make_move(self.board, action, marker)
        if self.is_winner(self.board, marker):
            reward, over = REWARD_WIN, True

        elif self.is_board_full():
            reward, over = REWARD_DRAW, True
        return self.board, reward, over

    @staticmethod
    def is_winner(board, marker):
        for combo in WINNING_COMBOS:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] == marker:
                return True
        return False

    @staticmethod
    def get_winning_combo(board):
        for combo in WINNING_COMBOS:
            if board[combo[0]] == board[combo[1]] == board[combo[2]]:
                return [combo[0], combo[1], combo[2]]
        return [None, None, None]

    @staticmethod
    def is_space_free(board, index):
        return board[index] == EMPTY_BOARD_SPACE

    def is_board_full(self):
        for i in range(1, 9):
            if self.is_space_free(self.board, i):
                return False
        return True

    @staticmethod
    def make_move(board, index, move):
        board[index] = move


def create_network(states, player):
    weights_layer_1 = tf.get_variable("W1_" + player, [BOARD_DIMENSIONS, 256],
                                      initializer=tf.random_normal_initializer(stddev=0.1))
    bias_layer_1 = tf.get_variable("b1_" + player, [256],
                                   initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(states, weights_layer_1) + bias_layer_1)
    # Not adding dropout for now
    # tf.nn.dropout(h1, keep_probability)

    weights_layer_2 = tf.get_variable("W2_" + player, [256, 64],
                                      initializer=tf.random_normal_initializer(stddev=0.1))
    bias_layer_2 = tf.get_variable("b2_" + player, [64],
                                   initializer=tf.constant_initializer(0))
    h2 = tf.nn.relu(tf.matmul(h1, weights_layer_2) + bias_layer_2)

    weights_output = tf.get_variable("Wo_" + player, [64, TOTAL_ACTIONS],
                                     initializer=tf.random_normal_initializer(stddev=0.1))
    bias_output = tf.get_variable("bo_" + player, [TOTAL_ACTIONS],
                                  initializer=tf.constant_initializer(0))

    network = tf.matmul(h2, weights_output) + bias_output
    return network


def get_optimizer():
    return tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DISCOUNT_RATE)


def input_valid(p2_action, board):
    return True if board[p2_action] == EMPTY_BOARD_SPACE else False


def get_players_stats():
    p1_lost = 0
    p1_won = 0
    p2_lost = 0
    p2_won = 0
    draw = 0
    return draw, p1_lost, p1_won, p2_lost, p2_won


def get_new_agent(agent_name, session, optimizer):
    return DeepAgent(agent_name,
                     session,
                     optimizer)


def train(session):
    env = Environment()
    fig = plt.figure()
    plot_figure = fig.add_subplot()
    p1_win_history, p2_win_history, iterations_count = list(), list(), list()

    optimizer = get_optimizer()
    summary_writer = tf.summary.FileWriter(LOGS_SAVE_LOCATION, session.graph)

    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Last 100 Episodes Average Episode Reward", episode_reward)
    summary_vars = [episode_reward]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    summary_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summaries = tf.summary.merge_all()

    player1 = get_new_agent("p1", session, optimizer)
    player2 = get_new_agent("p2", session, optimizer)

    saver = tf.train.Saver()
    session = load_model(session, saver)

    p1_playing_history = deque(maxlen=1000)
    p2_playing_history = deque(maxlen=1000)

    draw, p1_lost, p1_won, p2_lost, p2_won = get_players_stats()

    p1_reward, p2_reward = 0.0, 0.0
    for iteration in range(TRAINING_ITERATIONS):
        state = np.array(env.reset())
        done = False
        while not done:
            p1_state = state
            p1_action = player1.greedy_action(state[np.newaxis, :])
            next_state, p1_reward, done = env.step(p1_action, env.p1_marker)
            if not done:
                p2_state = np.array(next_state)
                p2_action = player2.greedy_action(state[np.newaxis, :])
                next_state, p2_reward, done = env.step(p2_action, env.p2_marker)

            p1win = True
            if done:
                if p2_reward == REWARD_WIN:
                    p1win = not p1win
                    p1_reward = REWARD_LOSE
                elif p1_reward == REWARD_WIN:
                    p2_reward = REWARD_LOSE
                if p2_reward == REWARD_DRAW or p1_reward == REWARD_DRAW:
                    p1_reward, p2_reward = REWARD_DRAW, REWARD_DRAW

            player1.update_model((p1_state, p1_action, p1_reward, next_state), done)
            player2.update_model((p2_state, p2_action, p2_reward, next_state), done)
            state = np.array(next_state)
            if done:
                if p1_reward == REWARD_DRAW:
                    draw += 1
                elif p1win:
                    p1_won += 1
                    p2_lost += 1
                elif not p1win:
                    p1_lost += 1
                    p2_won += 1
                p1_playing_history.append(p1_reward)
                p2_playing_history.append(p2_reward)
                break

        if iteration % 100 == 99:
            p1_mean_rewards = np.mean(p1_playing_history)
            print("Episode {}".format(iteration))
            print("P1 won:" + str(p1_won))
            print("P2 won:" + str(p2_won))
            print("draw:" + str(draw))
            session.run(summary_ops[0], feed_dict={summary_placeholders[0]: float(p1_mean_rewards)})
            result = session.run(summaries)
            summary_writer.add_summary(result, iteration)

            p1_win_history.append(p1_won)
            p2_win_history.append(p2_won)
            iterations_count.append(iteration)
            draw, p1_lost, p1_won, p2_lost, p2_won = get_players_stats()
            saver.save(session, SAVED_MODELS_LOCATION)

    # Plot the training graphs
    plot_figure.plot(iterations_count, p1_win_history, color='orange')
    plot_figure.plot(iterations_count, p2_win_history, color='blue')
    fig.show()


def load_model(session, saver):
    # Check for trained model: if present load into session
    checkpoint = tf.train.get_checkpoint_state(MODEL_FOLDER_NAME)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
    return session


def test(player1, player2, session):
    env = Environment()
    optimizer = get_optimizer()

    player1 = get_new_agent(player1, session, optimizer)

    if not player2 == "Human":
        player2 = get_new_agent(player2, session, optimizer)

    saver = tf.train.Saver()
    _ = load_model(session, saver)

    done = False
    state = np.array(env.reset())
    gui = GUI(state[np.newaxis, :][0])
    while not done:
        current_state = state[np.newaxis, :]
        player1_action = player1.greedy_action(current_state, False)
        print("P1 move: " + str(player1_action))
        next_state, player1_reward, done = env.step(player1_action, env.p1_marker)
        state = np.array(next_state)
        current_state = state[np.newaxis, :]
        gui.update(current_state[0], env.get_winning_combo(current_state[0]), done)
        if not done:
            if player2 == "Human":
                p2_action = int(input("Your Turn\n"))
                while not input_valid(p2_action, current_state[0]):
                    p2_action = int(input("No cheating allowed\n"))
            else:
                p2_action = player2.greedy_action(current_state, False)
            next_state, player2_reward, done = env.step(p2_action, env.p2_marker)
            print("P2 move: " + str(p2_action))
        state = np.array(next_state)
        current_state = state[np.newaxis, :]
        gui.update(current_state[0], env.get_winning_combo(current_state[0]), done)
    if player2_reward == REWARD_DRAW or player1_reward == REWARD_DRAW:
        print("Draw")
    elif player1_reward == REWARD_WIN:
        print("Player 1 Won")
    elif player2_reward == REWARD_WIN:
        print("Player 2 Won")
    else:
        print("Well played\nYou've beaten an ill-trained machine")
    # gui.app.destroy()


if __name__ == '__main__':
    tf.disable_eager_execution()
    sess = tf.Session()
    # Uncomment to train
    # train(sess)
    # There are two agents- So select either p1 or p2 as the agent name
    p1 = "p1"
    p2 = "p2"
    # Uncomment to play player vs player
    # test(p1, p2, sess)
    # Uncomment to play player vs human
    test(p1, "Human", sess)

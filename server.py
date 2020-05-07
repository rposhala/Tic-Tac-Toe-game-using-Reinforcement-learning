from flask import Flask, request
from flask import render_template
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

## function to check whether board is its terminal state
def check_winner(game):
    winner = ''
    checkfor = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for line in checkfor:
        s = str(game[line[0]]) + str(game[line[1]]) + str(game[line[2]])
        if s == 'XXX':
            winner = 'The winner is PLAYER 1'
            break
        elif s == 'OOO':
            winner = 'The winner is PLAYER 2'
            break
    if not any(a for a in game if type(a) != str) and 0 not in game and winner == '':
        winner = 'No winner, its a tie'
    
    
    return winner
  
#################
## class defined to initialize player instance 
class Sarsa_Agent:
    def __init__(self, name, exploration=0.5):
        self.name = name ## player name used to store policies specific to the player name
        self.terminate = None ## to store if it is in terminal state
        self.exploration_rate = exploration ## attribute which defines the exploration rate
        self.discounted_gamma = 0.7 ## decay gamma value which limits the rewards to the board states which are far from board terminal state
        self.learning_rate = 0.2 ## learning rate which Q-value adds up to convergence
        self.states = []  # to store all board states in a game
        self.dict_state_value = {}  # state of the board -> Q-value
        
    ## function to determine which step should be taken by the agent
    def NextStep(self, available_positions, current_board_state, turn):
        if np.random.uniform(0, 1) <= self.exploration_rate:
            idx = np.random.choice(len(available_positions)) # picks random next step from available positions
            action = available_positions[idx]
        else:
            value_max = -500
            ## picks the next step from the available postions based on the maximum Q-values
            for i in available_positions:
                next_board_state = current_board_state.copy()
                next_board_state[i] = turn
                next_boardHash = self.getHash(next_board_state)
                value = 0 if self.dict_state_value.get(next_boardHash) is None else self.dict_state_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = i
        # print("{} takes action {}".format(self.name, action))
        return action

    ## get the hash address of the board state
    def getHash(self, game):
        gameHash = str(game)
        return gameHash


    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # fucntion to compute and assign Q-values for each step (board state) at the end of game
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.dict_state_value.get(st) is None:
                self.dict_state_value[st] = 0
            ## SARSA Algorithm
            self.dict_state_value[st] += self.learning_rate * (self.discounted_gamma * reward - self.dict_state_value[st])
            reward = self.dict_state_value[st]

    ## fucntion to reset the player states
    def reset(self):
        self.states = []

    ## function to save the policies (it appends computed policies to existing policies if any exists)
    def savePolicy(self):
        if os.path.exists('policy_' + str(self.name)):
          obj = open('policy_' + str(self.name), 'rb')
          existing_policies = pickle.load(obj)
          for i in existing_policies.keys():
            self.dict_state_value[i] = existing_policies[i]
            
        write_file = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.dict_state_value, write_file)
        write_file.close()
    ## function to load the policies to any player instance
    def loadPolicy(self, file):
        read_file = open(file, 'rb')
        self.dict_state_value = pickle.load(read_file)
        read_file.close()



#################

## get the hash address of the board state
def getHash(game):
  return str(game)

### function which gives rewards for each player at the end of each game
def reward(player1,player2,winner):
  if winner == 'The winner is PLAYER 1':
    player1.feedReward(1)
    player2.feedReward(0)
  elif winner == 'The winner is PLAYER 2':
    player1.feedReward(0)
    player2.feedReward(1)
  else :
    player1.feedReward(0.1)
    player2.feedReward(0.5)

######## function to train the agent ###########

def agent_training(player1,player2,rounds):
	p1 = 0
	p2 = 0
	tie = 0
	p1_plot = 0
	p2_plot = 0
	tie_plot = 0
	# plot_p1 = []
	# plot_p2 = []
	# iterations_count =[]
	# plot_tie = []
	for i in range(rounds):
		w = ''
		s = list(range(9))
		sav = np.zeros(9)
		avail = s.copy()
		a = s.copy()
		turn = ''
		while(w == ''):
			# Player 1 turn to play
			turn = 'X'
			p1_action = player1.NextStep(a, sav, 1)
			sav[p1_action] = 1
			s[p1_action] = turn
			a.remove(p1_action)
			board_hash = getHash(sav)
			player1.addState(board_hash)
			win = check_winner(s)
			if win != '':
				if win == 'No winner, its a tie':
					tie+=1
					tie_plot += 1
				else :
					p1+=1
					p1_plot+=1
				reward(player1,player2,win)
				player1.reset()
				player2.reset()
				break
			else:
	      	# Player 2 turn to play
				turn = 'O'
				p2_action = player1.NextStep(a, sav, -1)
				sav[p2_action] = -1
				s[p2_action] = turn
				a.remove(p2_action)
				board_hash = getHash(sav)
				player2.addState(board_hash)
				win = check_winner(s)
				if win != '':
					if win == 'No winner, its a tie':
						tie+=1
						tie_plot += 1
					else :
						p2+=1
						p2_plot+=1
					reward(player1,player2,win)
					player1.reset()
					player2.reset()
					break
		'''
		## unblock to look at the plots and stats
		if i%500 == 0:
			iterations_count.append(i)
			plot_p1.append(p1_plot)
			plot_p2.append(p2_plot)
			plot_tie.append(tie_plot)
			p1_plot = 0
			p2_plot = 0
			tie_plot = 0

	fig = plt.figure()
	plot_figure = fig.add_subplot()
	plot_figure.plot(iterations_count, plot_p1, color='orange')
	plot_figure.plot(iterations_count, plot_p2, color='blue')
	plot_figure.plot(iterations_count, plot_tie, color='green')
	fig.show()
	print("Total iterations: ",rounds)
	print("Player1 (X) wins: ",p1)
	print("Player2 (O) wins: ",p2)
	print("Tie: ",tie)
'''
######## End of function to train the agent ###########

'''
####### Unblock to train the Player1 and Player2 Agents ########
##### Agent at maximum difficulty (hard) level #####
hplayer1 = Sarsa_Agent("hard_p1")
hplayer2 = Sarsa_Agent("hard_p2")
print("training for hard level")
agent_training(hplayer1, hplayer2,50000)
print("done")
print("- - - - - - - - - - - - - - - - -")
hplayer1.savePolicy()
hplayer2.savePolicy()
######################################################

##### Agent at moderate difficulty (medium) level #####
mplayer1 = Sarsa_Agent("medium_p1")
mplayer2 = Sarsa_Agent("medium_p2")
print("training for medium level")
agent_training(mplayer1, mplayer2,3000)
print("done")
print("- - - - - - - - - - - - - - - - -")
mplayer1.savePolicy()
mplayer2.savePolicy()
######################################################

##### Agent at little difficulty (easy) level #####
eplayer1 = Sarsa_Agent("easy_p1")
eplayer2 = Sarsa_Agent("easy_p2")
print("training for easy level")
agent_training(eplayer1, eplayer2,100)
print("done")
eplayer1.savePolicy()
eplayer2.savePolicy()
######################################################
################################################################
'''


playerselect = 'X'
difficulty = 'easy'
board = 0
pos = 0
sav = 0
player1 = 0
player2 = 0
# player1 = Sarsa_Agent("agent", exploration= 0)
# player2 = Sarsa_Agent("agent", exploration= 0)


whichplayer = 1
w = ''

@app.route("/")
def init():
    return render_template('tictactoe.html')

@app.route("/start",methods = ['POST', 'GET'])
def start():
	global playerselect
	playerselect = request.form.get('userselect') # did player choose X or O
	difficulty = request.form.get('difficulty') # easy, medium, hard
	place = -1
	global board
	global sav
	global pos
	global player1
	global player2
	board = list(range(9))
	pos = board.copy()
	sav = np.zeros(9)
	# player1 = Player("agent", exp_rate = 0)
	# player1.loadPolicy("policy_hplayer1")
	# print(difficulty)
	if difficulty == 'easy':
		policy_p1 = "policy_easy_p1"
		policy_p2 = "policy_easy_p2"
	elif difficulty == 'medium':
		policy_p1 = "policy_medium_p1"
		policy_p2 = "policy_medium_p2"
	elif difficulty == 'hard':
		policy_p1 = "policy_hard_p1"
		policy_p2 = "policy_hard_p2"
	player1 = Sarsa_Agent("agent", exploration= 0)
	player2 = Sarsa_Agent("agent", exploration= 0)
	player1.loadPolicy(policy_p1)
	player2.loadPolicy(policy_p2)
	if playerselect == 'O':
		place =  player1.NextStep(pos,sav,1)  # which place system is keeping
		board[place] = 'X'
		sav[place] = 1
		pos.remove(place)
	return str(place)

@app.route("/getPlace",methods = ['POST', 'GET'])
def getPlace():
	userinput = request.form.get('userinput') # which place user has clicked
	global playerselect
	global board
	global sav
	global pos
	global player1
	global player2
	index = int(userinput)
	agent = 'X' if playerselect == 'O' else 'O'
	agent_number = 1 if playerselect == 'O' else -1
	human_number = -1 if playerselect == 'O' else 1
	player = player1 if playerselect == 'O' else player2
	board[index] = playerselect
	sav[index] = human_number
	pos.remove(index)
	w = check_winner(board)
	# print(w)
	w = 'Human wins!' if w != '' and w != 'No winner, its a tie' else w
	place = -1
	if w == '' :
		place = player.NextStep(pos,sav,agent_number) # which place system is keeping
		board[place] = agent
		sav[place] = agent_number
		pos.remove(place)
		w = check_winner(board)
		w = 'Agent wins!' if w != '' and w != 'No winner, its a tie' else w
	
	resp = {}
	resp['place'] = str(place)
	resp['w'] = w
	return resp

if __name__ == "__main__":
    app.run()
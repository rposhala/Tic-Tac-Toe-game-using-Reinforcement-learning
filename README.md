# Tic Tac Toe using Reinforcement learning

Tic Tac Toe game was designed an deployed on AWS server. In the back-end, two agents were trained (one as player 1 and other as player 2) to play the game Tic Tac Toe using Reinforcement learning  algorithm: SARSA. After certain epochs, agents get trained to a certain level where no agent will lose (it would be a tie in most of the cases). 

Agents are trained and their policies are saved at three intervals during the training after certain epochs. And these saved policies are loaded and used as 'Easy', 'Medium' and 'Hard' levels for each of agent (representing each player). These agents from the back-end and integrated to front-end using flask API to let user play with the agent as either player 1 or player 2 with any level of difficulty (easy or medium or hard). 
This project was deployed on AWS server. GitHub repository has all files related to the game and clear instructions to design and deploy the game. 
Playing with the agent as player 2 or player 1 with difficulty level 'Hard', we can at most tie the game. In most of the cases, agent wins.

**Game has been deployed on AWS server, below is the link for the game:

http://18.223.101.138:8080/

Policies of both the players of each level are saved when the agents are trained previously so that each time the end user plays the game accessing the above link with no need to train the agent again and again.

**Running the app in the local system:

Submitted saved policies, python source code file with name server.py along with a folder named template which has front end code with name tictactoe.html (basic structure of file placement for flask api to integrate both front and back end) should be placed in the same folder. And from terminal we need to execute the python code (server.py) from the above folder using command: 
<b>python server.py</b> 
It works for Mac.

Once the application starts, the Tic Tac Toe game can be accessed in the local system using http://localhost:5000/ link.



The DQN can be run using <b>python game.py</b>

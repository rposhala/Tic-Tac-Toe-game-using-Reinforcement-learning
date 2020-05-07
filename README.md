# Machine_learning

Game has been deployed on AWS server, below is the link for the game:

http://18.223.101.138:8080/

Policies of both the players of each level are saved when the agents are trained previously so that each time the end user plays the game accessing the above link with no need to train the agent again and again.

Running the app in the local system:

Submitted saved policies, python source code file with name server.py along with a folder named template which has front end code with name tictactoe.html (basic structure of file placement for flask api to integrate both front and back end) should be placed in the same folder. And from terminal we need to execute the python code (server.py) from the above folder using command: 
<b>python server.py</b> 
It works for Mac.

Once the application starts, the Tic Tac Toe game can be accessed in the local system using http://localhost:5000/ link.



The DQN can be run using <b>python game.py</b>

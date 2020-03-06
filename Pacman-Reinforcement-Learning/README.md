# Pacman Search Algorithm

## Task

Your task Your tasks relate to the assignment at [https://inst.eecs.berkeley.edu/~cs188/fa18/
project1.html.]( https://inst.eecs.berkeley.edu/~cs188/fa18/
project1.html.)

## Part(1)

Implement the Iterative Deepening Search algorithm discussed in lectures. You should be able to test the algorithm using the following command:

```python
 python pacman.py -l mediumMaze -p SearchAgent -a fn=ids
```

Other layouts are available in the layouts directory, and you can easily create you own!



## Part (2)

Implement the Weighted A* algorithm discussed in lectures using W = 2. You may hardcode this weight into your algorithm (that is, do not pass as a parameter). You should be able to call your function using the **fn=wastar** parameter from the command line, i.e. you should be able to test the algorithm using the following command:

```python
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=wastar,heuristic=manhattanHeuristic
```



## Part (3)

Now we wish to solve a more complicated problem. Just like in Q7 of the Berkerley Pac Man framework, we woud like to create an agent that will eat all of the dots in a maze. Before doing so, however, the agent must eat a Capsule that is present in the maze. Your code should ensure that no food is eaten before the Capsule. You can assume that there is always exactly one Capsule in the maze, and that there will always be at least one path from Pacman’s starting point to the capsule that doesn’t pass through any food.

In order to implement this, you should create a new problem called **CapsuleSearchProblem** and a new agent called **CapsuleSearchAgent.** You will also need to implement a suitable **foodHeuristic**. You may choose to implement other helper classes/functions. You should be able to test your program by running the following code:



```python
python pacman.py -l capsuleSearch -p CapsuleSearchAgent -a fn=wastar,prob=CapsuleSearchProblem,heuristic=foodHeuristic
```

An agent that eats the capsule then proceeds to eat all of the food on the maze will receive 3 marks. The remaining 3 marks will be based on the performance of your agent (i.e. number of nodes expanded), as in Q7 of the Berkeley problem. Since you are using the Weighted A* algorithm, however, the number of node expansions required for each grade will vary.
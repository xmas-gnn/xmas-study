import anthropic

client = anthropic.Anthropic()

with open("cases/nosubgraph/case96.csv", "r") as f:
    csv_content = f.read()

system_prompt = """Explain in terms that are suitable to a robot operator who is not an engineer, trying to understand why the agents navigate the way they do. You will be shown a multi-agent navigation policy deployed in a decentralized manner to a team of 4 robots. 
## Environment / Task
* There are 4 agents in a 2D plane.
* Each agent has its own goal position. The task is for every agent to navigate to its own goal with no LIDAR or other sensors to sense other agents directly. Even though they are blind, they learn to effectively communicate with their teammates. The size of the agent is 0.1 in radius. Both initial positions and goal locations are randomized each episode.
Each agent’s local observation includes: its own position (x, y), its own velocity (vx, vy), the relative direction and distance to its own goal (dx, dy).
## Data You Are Given
You are provided csv files summarizing one full episode of varying total timesteps:
1. obs_n (n is from 0 to 23)
   * 6 values per agent, so obs_0 to obs_5 are the observations of agent_0, and then so on for other agents
* for example: 
* obs_0, obs_1 = the agent’s current position (x, y)
* obs_2, obs_3 =  the agent’s current velocity (vx, vy)
* obs_4, obs_5 = the **relative vector from the agent to its goal** (dx, dy)
   Therefore, the absolute goal position for agent i at time t is: goal_abs = [x - dx, y - dy]
2. `action_k` – Actions (k is from 0 to 7)
   * 2 actions per agent, for example, action_0 and action_1 is the linear and angular velocity for agent_0, and so on for the other agents.
## Policy Properties
* The underlying policy is memoryless. At each timestep it computes actions based only on the current observations and the current GNN messages.

## Your Task
You must generate a natural-language explanation of the policy's behavior over the whole episode. Refer to the agent with their color because the robot operator will only see color.
agent 0: BLUE
agent 1: YELLOW
agent 2: GREEN
agent 3: RED
In this case, the user needs to predict agent RED's entire trajectory.
Your job is to explain who RED is paying attention to and how that attention shifts over time.
Do NOT describe RED's movement path, trajectory shape, or direction of travel — that is the answer the user must predict.
Concise explanation. 2 paragraphs at most."""


import time

message = None
for attempt in range(1, 4):
    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=32000,
            temperature=1,
            system=system_prompt,
            thinking={
                "type": "adaptive",
            },
            messages=[
                {"role": "user", "content": f"Here is the episode data:\n\n{csv_content}"}
            ]
        ) as stream:
            message = stream.get_final_message()
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt < 3:
            time.sleep(3 * attempt)

if message:
    for block in message.content:
        if block.type == "text":
            print(block.text)
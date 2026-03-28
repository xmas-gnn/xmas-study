import anthropic

client = anthropic.Anthropic(api_key="replace-with0real")

with open("case1.csv", "r") as f:
    csv_content = f.read()

system_prompt = """Explain in terms that are suitable to a robot operator who is not an engineer, trying to understand why the agents navigate the way they do. You will be shown a multi-agent navigation policy deployed in a decentralized manner to a team of 4 robots.
## Environment / Task
* There are 4 agents in a 2D plane.
* Each agent has its own goal position. The task is for every agent to navigate to its own goal with no LIDAR or other sensors to sense other agents directly. Even though they are blind, they learn to effectively communicate with their teammates. The size of the agent is 0.1 in radius. Both initial positions and goal locations are randomized each episode.
Each agent's local observation includes: its own position (x, y), its own velocity (vx, vy), the relative direction and distance to its own goal (dx, dy).
## Data You Are Given
You are provided csv files summarizing one full episode of varying total timesteps:
1. obs_n (n is from 0 to 23)
   * 6 values per agent, so obs_0 to obs_5 are the observations of agent_0, and then so on for other agents
* for example: 
* obs_0, obs_1 = the agent's current position (x, y)
* obs_2, obs_3 =  the agent's current velocity (vx, vy)
* obs_4, obs_5 = the relative vector from the agent to its goal (dx, dy)
   Therefore, the absolute goal position for agent i at time t is: goal_abs = [x - dx, y - dy]
2. action_k – Actions (k is from 0 to 7)
   * 2 actions per agent, for example, action_0 and action_1 is the linear and angular velocity for agent_0, and so on for the other agents.
3. edge_mask_j – Edge importance over time (j is from 0 to 11)
   * There are 3 edges per agent. There is no self loop. There are 12 total edges for 4 agents.
* For example, edge_mask_0, edge_mask_1, and edge_mask_2 belongs to agent_0 and represents the connection to agent_1, agent_2, and agent_3 respectively
   * These edge importances are post-hoc estimates from Attention Explainer. It is an explainer that uses the attention coefficients produced by an attention-based GNN as edge explanation.
* The policy is trained in model-free RL (PPO) and it is structured as a Graph Neural Network. The reason to use GNN is its permutation invariance property, the latent message for each agent is processed through Message passing - each agent aggregates information from its neighbors along edges.
* Here, an edge's importance tells us how much the information along that edge contributed to the decision for the receiving agent at that timestep.
## Policy Properties
* The underlying policy is memoryless. At each timestep it computes actions based only on the current observations and the current GNN messages.

## Your Task
You must generate a natural-language explanation of the policy's behavior over the whole episode. Refer to the agent with their color because the robot operator will only see color.
agent 0: BLUE
agent 1: YELLOW
agent 2: GREEN
agent 3: RED
The user needs to predict where agent green will reach its goal locaiton. 
Your job is to explain what agent green is thinking without spilling the answer to the user. 
The user need to figure it out from you explanation.
Concise explanation. 2 paragraphs at most"""

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=4096,
    temperature=0.2,
    system=system_prompt,
    messages=[
        {"role": "user", "content": f"Here is the episode data:\n\n{csv_content}"}
    ]
)

print(message.content[0].text)
import anthropic

client = anthropic.Anthropic()

with open("cases/subgraph/case90.csv", "r") as f:
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
3. `edge_mask_j` – Edge importance over time (j is from 0 to 11)
   * There are 3 edges per agent. There is no self loop. There are 12 total edges for 4 agents.
* For example, edge_mask_0, edge_mask_1, and edge_mask_2 belongs to agent_0 and represents the connection to agent_1, agent_2, and agent_3 respectively
   * These edge importances are post-hoc estimates from Attention Explainer. It is an explainer that uses the attention coefficients produced by an attention-based GNN as edge explanation. 
* The policy is trained in model-free RL (PPO) and it is structured as a Graph Neural Network. The reason to use GNN is its permutation invariance property, the latent message for each agent is processed through Message passing - each agent aggregates information from its neighbors along edges.
* Here, an edge’s importance tells us how much the information along that edge contributed to the decision for the receiving agent at that timestep.
## Policy Properties
* The underlying policy is memoryless. At each timestep it computes actions based only on the current observations and the current GNN messages.
## Your Task
You must generate a natural-language explanation of the policy’s behavior over the whole episode. Refer to the agent with their color because the robot operator will only see color.
agent 0: BLUE
agent 1: YELLOW
agent 2: GREEN
agent 3: RED
### Focus of the explanation
For each agent (0, 1, 2, 3):
1. Describe the basic trajectory: where it starts and its goal is number
* Rough **path style** over time:
     * mostly straight vs gently curved vs zig-zag,
     * whether it seems to take detours or side-steps
2. Describe how much this agent “cares about” other agents, only using this agent’s outgoing edge importances
   * Explain in plain language, e.g.: At the first half of the trajectory, Agent 0’s decision strongly depends on Agent 2, suggesting it adjusts its move based on where Agent 2 is going. Later, Agent 0 briefly pays a lot of attention to Agent 3, likely to avoid crossing its path.
3. Connect edge importance to behavior:
   * When you see a spike in an outgoing edge from agent A to agent B: Interpret it as: Agent A is heavily using information from Agent B right now to decide what to do.
   * Connect this to the motion:
     * Is the agent slowing down, turning, or taking a curved path around that time?
     * Does it look like it’s avoiding being in the same spot as that neighbor?
4. Acknowledge explainer uncertainty:
   * When you talk about edge importance, Note that
     * These are post-hoc explainer estimates.
     * A high value means this communication channel likely influenced the decision more, but there may be some noise.
### Style / Tone Requirements
* Write as if explaining to intelligent non-experts:
  * Avoid heavy math or jargon.
  * If you use words like “GNN,” “edge importance,” or “message passing,” immediately translate them into a simple idea
 * Use clear, concrete storytelling
  * You do not need to describe all timesteps one by one.
Here is an example format to follow:
“Agent 0
Starts near the top-middle and curves left then down to reach its lower-left goal. Early on it watches Agent 1 a lot, then mostly tracks Agent 3 (and a bit of Agent 2 near the end) to slide around others instead of going straight through them.”
The total answer should be within 200 words
Do not explain “how much others care about this agent.”. Focus only on how much each agent cares about others.
"""

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

# message = client.messages.create(
#     model="claude-opus-4-6",
#     max_tokens=4096,
#     temperature=0.2,
#     system=system_prompt,
#     messages=[
#         {"role": "user", "content": f"Here is the episode data:\n\n{csv_content}"}
#     ]
# )

# print(message.content[0].text)
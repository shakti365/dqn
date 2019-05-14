from agent import Agent

render = True
memory_capacity = 1000
agent = Agent(render=render, model=None, memory_capacity=memory_capacity)
agent.run_episode(num_episodes=5)
print (len(agent.memory.states))
print (max(agent.memory.episodes))

# Initialize replay memory to capcity N

# Initialize Q network parameters
# Copy Q network weights to target network weights

# For episode:

    # Initialize a state list s

    # Observe an input x and push to state list s

    # Get state from processing the state list s

    # For step:

        # Get recommended action from the policy

        # Observe next state x and reward r

        # Push next state to state list

        # Get next state from processing the state list

        # Push transition to replay memory

        # Sample minbatch from replay memory

        # Optimize TD loss

        # Copy weights every 10K steps

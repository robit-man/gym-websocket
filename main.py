from fastapi import FastAPI, WebSocket
import gymnasium as gym
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            # Process received data
            env = gym.make('CartPole-v1')
            observation = env.reset()
            response_data = []
            for _ in range(1000):
                action = env.action_space.sample()  # Replace with your RL model's action
                # Ensure correct number of values are unpacked
                result = env.step(action)
                print(result)  # Print result to verify the number of values
                if len(result) == 4:
                    observation, reward, done, info = result
                else:
                    observation, reward, done, truncated, info = result
                    done = done or truncated  # Ensure 'done' status is handled correctly
                response_data.append({
                    "observation": observation.tolist(),
                    "reward": reward,
                    "done": done
                })
                if done:
                    observation = env.reset()
            env.close()
            await websocket.send_json({"result": response_data})
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            break
    await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)

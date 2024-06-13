from fastapi import FastAPI, WebSocket
import uvicorn
import gymnasium as gym
from stable_baselines3 import PPO
import custom_bipedal_walker  # Import the custom environment to ensure it's registered
import torch
import os

app = FastAPI()

# Load the trained model
model_path = "ppo_custom_bipedal_walker.zip"

if os.path.exists(model_path):
    model = PPO.load(model_path)
else:
    model = None
    print(f"Model file {model_path} not found. Please train the model first.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        if model is None:
            await websocket.send_json({"error": "Model not loaded. Please train the model first."})
            return

        # Process received data
        env = gym.make('CustomBipedalWalker-v0')
        observation, _ = env.reset()
        response_data = []
        for _ in range(1000):
            action, _states = model.predict(observation)
            observation, reward, done, truncated, info = env.step(action)
            response_data.append({
                "observation": observation.tolist(),
                "done": done
            })
            if done:
                observation, _ = env.reset()
        env.close()
        # Log the response data
        print("Sending response:", response_data)
        await websocket.send_json({"result": response_data})
    except Exception as e:
        error_message = {"error": str(e)}
        await websocket.send_json(error_message)
        print(f"Error occurred: {e}")
    finally:
        await websocket.close()

@app.get("/load_weights")
async def load_weights():
    try:
        global model
        model = PPO.load("ppo_custom_bipedal_walker")
        return {"status": "Weights loaded successfully"}
    except Exception as e:
        return {"status": "Failed to load weights", "error": str(e)}

@app.get("/export_weights")
async def export_weights():
    try:
        torch.save(model.policy.state_dict(), "ppo_custom_bipedal_walker_weights.pth")
        return {"status": "Weights exported successfully"}
    except Exception as e:
        return {"status": "Failed to export weights", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

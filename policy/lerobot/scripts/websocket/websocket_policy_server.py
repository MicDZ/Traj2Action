import asyncio
import logging
import traceback
from typing import Dict, Any, Optional

import torch
import numpy as np
import einops
import websockets.asyncio.server
import websockets.frames

try:
    from openpi_client import msgpack_numpy
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    import json

from lerobot.common.policies.pretrained import PreTrainedPolicy


class WebsocketPolicyServer:
    """WebSocket policy server, compatible with openpi_client's WebsocketClientPolicy"""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        device: str = "cuda",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._device = device
        self._metadata = metadata or {}
        
        # Set logging level
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Start server"""
        asyncio.run(self.run())

    async def run(self):
        """Run async server"""
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info(f"WebSocket server started on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        """Handle WebSocket connection"""
        logging.info(f"Connection from {websocket.remote_address} opened")
        
        # Use msgpack_numpy for serialization
        if HAS_MSGPACK:
            packer = msgpack_numpy.Packer()
            await websocket.send(packer.pack(self._metadata))
        else:
            await websocket.send(json.dumps(self._metadata))

        while True:
            try:
                # Receive and parse data
                raw_data = await websocket.recv()
                if HAS_MSGPACK:
                    obs = msgpack_numpy.unpackb(raw_data)
                else:
                    obs = json.loads(raw_data)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Process input data
                    observation = {}
                    
                    # Process image data
                    if "observation/wrist_image" in obs:
                        wrist_img = torch.from_numpy(np.array(obs["observation/wrist_image"])).float()
                        wrist_img = einops.rearrange(wrist_img, "h w c -> 1 c h w") / 255.0
                        observation["wrist_image"] = wrist_img.to(self._device)
                        print("wrist_image")
                        
                    if "observation/top_image" in obs:
                        top_img = torch.from_numpy(np.array(obs["observation/top_image"])).float()
                        top_img = einops.rearrange(top_img, "h w c -> 1 c h w") / 255.0
                        observation["top_image"] = top_img.to(self._device)

                    if "observation/main_image" in obs:
                        main_img = torch.from_numpy(np.array(obs["observation/main_image"])).float()
                        main_img = einops.rearrange(main_img, "h w c -> 1 c h w") / 255.0
                        observation["main_image"] = main_img.to(self._device)

                    if "observation/human_image" in obs:
                        human_img = torch.from_numpy(np.array(obs["observation/human_image"])).float()
                        human_img = einops.rearrange(human_img, "h w c -> 1 c h w") / 255.0
                        observation["human_image"] = human_img.to(self._device)
                        
                    # Process state data
                    if "observation/state" in obs:
                        state = torch.from_numpy(np.array(obs["observation/state"])).float()
                        state = state.unsqueeze(0)  # Add batch dimension
                        observation["state"] = state.to(self._device)
                    
                    if "observation/state_trajectory" in obs:
                        state_traj = torch.from_numpy(np.array(obs["observation/state_trajectory"])).float()
                        state_traj = state_traj.unsqueeze(0)  # Add batch dimension
                        observation["state_trajectory"] = state_traj.to(self._device)
                        
                    # Process prompt
                    if "prompt" in obs:
                        observation["task"] = [obs["prompt"]]
                    
                    # Execute model inference
                    with torch.inference_mode():
                        trajectory, action = self._policy.select_action(observation)
                    
                    # Process output
                    if isinstance(action, torch.Tensor):
                        trajectory = trajectory.cpu().numpy()
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    result = {"actions": action.tolist() if hasattr(action, 'tolist') else action, 
                              "trajectory": trajectory.tolist() if hasattr(trajectory, 'tolist') else trajectory}
                
                # Send result
                if HAS_MSGPACK:
                    await websocket.send(packer.pack(result))
                else:
                    await websocket.send(json.dumps(result))
                    
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logging.error(f"Error in handler: {e}")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                break

    def _json_serializer(self, obj):
        """JSON serialization helper function"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return str(obj)

import numpy as np    
import PIL.Image as Image
from inference_utils import resize_with_pad, convert_to_uint8, WebsocketClientPolicy

if __name__ == "__main__":
    ##### TODO: Modify the host and port according to the actual policy server. #####
    client = WebsocketClientPolicy(
            host="localhost",  #### Replace with the actual host
            port=8077
        )

    human_image = Image.open("assets/human_camera_video.jpg")
    main_image = Image.open("assets/main_camera_video.jpg")
    wrist_image = Image.open("assets/wrist_camera_video.jpg")
    top_image = Image.open("assets/top_camera_video.jpg")

    main_image_np = np.array(main_image)
    wrist_image_np = np.array(wrist_image)
    human_image_np = np.array(human_image)
    top_image_np = np.array(top_image)


    ###### TODO: Modify the state and images according to the acutal franka robot. #####
    state = np.array([0.40896203, 0.15705565, 0.02108654])
    state_traj = np.array([0.40896203, 0.15705565, 0.02108654])
    ####################################################################################


    ###### HACK: Prompt can be sent from the client. If no prompt is provided, the prompt will be determined from the server. #####
    
    # prompt = "put both the alphabet soup and the tomato sauce in the basket"
    prompt = "pick up the tomato"

    element = {
                "observation/main_image": main_image_np,
                "observation/wrist_image": wrist_image_np,
                "observation/human_image": human_image_np,
                "observation/top_image": top_image_np,
                "observation/state": state,
                "observation/state_trajectory": state_traj,
                "prompt": prompt,
            }

    action_chunk = client.infer(element)["actions"]
    trajectory_chunk = client.infer(element)["trajectory"]

    print(f'trajectory:{trajectory_chunk}')
    if action_chunk is not None:
        print("action_chunk:", [action[-1] for action in action_chunk])


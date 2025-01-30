import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from datetime import datetime

def test_qwen_model():
        model_path = "/home/dbisai/LLM/"
        #model_name = "Qwen2.5-3B"
        #model_name = "Qwen2.5-0.5B"
        #model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
        #model_name = "DeepSeek-R1-Distill-Qwen-7B"
        model_name = "DeepSeek-R1-Distill-Llama-8B"

        model_namepath = model_path + model_name

        # Determine the device
        # WITH CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # WITH DIRECTML
        #device = torch_directml.device()
        print(f"\n[INFO]: Device used: {device}")

        # Load the tokenizer and model
        print(f"\n[INFO]: Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_namepath)
        model = AutoModelForCausalLM.from_pretrained(model_namepath, device_map="auto", torch_dtype="auto").to(device)
        print(f"\n[INFO]: Model {model_name} loaded successfully!")

        # Ensure the logs directory exists
        os.makedirs("logs", exist_ok=True)

        for i in range(4):
                print(f"\n[INFO]: Starting timer...")
                start_time = time.time()
                prompt = ""

                match i:
                        case 0:
                                # Simple prompt
                                prompt = "What is the purpose of error codes in robotics?"
                                print(f"\n[PROMPT]:\n {prompt}\n")
                        case 1:
                                # Prompt with data
                                test_error = {
                                        "Error-Code": "E105",
                                        "Error-Message": "Temperature Threshold Exceeded",
                                        "Error-Details": "The internal temperature of the robotic arm's motor exceeded 85°C, which is beyond the safe operating range.",
                                        "Error-Time": "2025-01-06T14:35:00Z",
                                        "Type of Error": "Hardware/Machine-Error",
                                        "Device-Type & Name": "Robotic Arm - RA2000"
                                        }
                                print(f"\n[ERROR]: Error Information:\n{test_error}")

                                prompt = f"You are an assistant for diagnosing robotic errors. Analyze the error based on the provided information, explain the cause of the issue, and suggest steps to resolve it.\n" + \
                                        f"Error Information:\n" + \
                                        f"{test_error}\n" + \
                                        f"Based on this information:\n" + \
                                        f"1. Provide a brief explanation of the error. [ERROR EXPLANATION]\n" + \
                                        f"2. Suggest possible causes. [POSSIBLE CAUSES]\n" + \
                                        f"3. Recommend steps to resolve the issue. [RECOMMENDED STEPS]\n"
                                #prompt = f"You are an assistant for diagnosing robotic errors. Analyze the error based on the provided information, explain the cause of the issue, and suggest steps to resolve it.\n" + \
                                        # f"Error Information:\n" + \
                                        # f"{test_error}\n" + \
                                        # f"\n" + \
                                        # f"Based on this information:\n" + \
                                        # f"1. Provide a brief explanation of the error.\n" + \
                                        # f"2. Suggest possible causes.\n" + \
                                        # f"3. Recommend steps to resolve the issue.\n" + \
                                        # f"\n" + \
                                        # f"Format your response like this:\n" + \
                                        # f"---\n" + \
                                        # f"**Error Explanation**:\n" + \
                                        # f"[Explanation]\n" + \
                                        # f"\n" + \
                                        # f"**Possible Causes**:\n" + \
                                        # f"- [Cause 1]\n" + \
                                        # f"- [Cause 2]\n" + \
                                        # f"- ...\n" + \
                                        # f"\n" + \
                                        # f"**Recommended Steps**:\n" + \
                                        # f"1. [Step 1]\n" + \
                                        # f"2. [Step 2]\n" + \
                                        # f"3. ...\n" + \
                                        # f"---"
                                print(f"\n[PROMPT]:\n {prompt}\n")
                        case 2:
                                # Simple prompt
                                prompt = "Levi is my father, Grisha is Levi's brother and Eren is Grisha's son. Who is Eren to me ?"
                                print(f"\n[PROMPT]:\n {prompt}\n")
                        case 3:
                                prompt = "Summmarize this website for me: https://github.com/deepseek-ai/DeepSeek-R1, and explain your thought process."
                                print(f"\n[PROMPT]:\n {prompt}\n")

                # Generate response
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_length=1000)

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)#.replace(prompt, "").strip()
                print(f"\n[ANSWER]:\n" + response)

                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"\n[INFO]: Promt executed and response generated in {execution_time:.2f} seconds") 

                # Write to log file
                print(f"\n[INFO]: Writing to log file...")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_filename = f"./logs/prompt_{timestamp}.log"
                with open(log_filename, "w") as log_file:
                        log_file.write(f"\nModel: {model_name}")
                        log_file.write(f"\nDevice: {device}\n\n------------\n\n")
                        log_file.write(f"Prompt:\n{prompt}\n\n------------\n\n")
                        log_file.write(f"Answer:\n{response}\n\n------------\n\n")
                        log_file.write(f"Execution time: {execution_time} seconds\n\n")
                
                print(f"\n[INFO]: Log file written successfully: {log_filename}")
                time.sleep(1)

if __name__ == "__main__":
    test_qwen_model()

from transformers import pipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import torch


print("Is CUDA available? ", torch.cuda.is_available())
print("Number of available GPUs: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name: ", torch.cuda.get_device_name(0))
else:
    print("No CUDA devices found.")


def test_libraries():
    print("Libraries imported successfully!")
    # Check if a GPU is available and use it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    summarizer = pipeline("summarization", device=device)
    print(f"Summarizer pipeline created on {'GPU' if device == 0 else 'CPU'}.")


if __name__ == "__main__":
    test_libraries()

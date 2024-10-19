import dspy
import logging
import random
from dspy import InputField, OutputField, Signature
from problem import Problem
from vor2 import Desc2PlanGenerator, Plan2CodeGenerator, ReviseCode, RevisePlan

# Configure logging with random colors
def get_random_color():
    return random.choice(['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m'])

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = get_random_color()
        reset = '\033[0m'
        return f"{color}{super().format(record)}{reset}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - Line %(lineno)d: %(message)s'))
logger.addHandler(handler)

def extract_code(response: str) -> str:
    # extract all text between <code> and </code> tags
    # there might be multiple code blocks, return the first one
    logger.info(f"Extracting code from response: {response}")
    import re
    code_blocks = re.findall(r'<code>(.*?)</code>', response, re.DOTALL)
    out =  code_blocks[-1] if code_blocks else ""
    if out == "":
        code_blocks = re.findall(r'```cpp(.*?)```', response, re.DOTALL)
        out = code_blocks[-1] if code_blocks else ""
    return out
class Agent:
    def __init__(self):
        self.desc2plan = Desc2PlanGenerator()
        self.plan2code = Plan2CodeGenerator()
        self.revise_code = ReviseCode()
        self.revise_plan = RevisePlan()

    def get_plan(self, desc: str, guidelines :str =""):
        response = self.desc2plan(
            problem_description=desc, 
            guidelines=guidelines
        )
        return response.plan

    def get_code(self, plan: str, input_ouput_format: str):
        response = self.plan2code(
            plan=plan,
            input_ouput_format=input_ouput_format
        )
        return response.cpp_program
    
    def revise_code(self, plan, broken_code, error, input_output_format):
        response = self.revise_code(
            plan=plan,
            broken_code=broken_code,
            error=error,
            input_output_format=input_output_format
        )
        return response


    def revise_plan(self, plan, problem_description, error):
        response = self.revise_plan(
            plan=plan,
            problem_description=problem_description,
            error=error,
        )
        return response

if __name__ == "__main__":
    problem_name = "Walk the Line"
    with open(f"Hacker cup/{problem_name}/statement.txt", "r") as f:
        desc = f.read()
    problem = Problem(
        desc=desc, 
        sample_in_file=f"Hacker cup/{problem_name}/sample_in.txt", 
        sample_out_file=f"Hacker cup/{problem_name}/sample_out.txt"
    )

    lm = dspy.Together(
        # model="meta-llama/Llama-3-70b-chat-hf", # Note: didn't find much a difference btwn mini & full gpt-4o
        model="google/gemma-2-27b-it", # Note: didn't find much a difference btwn mini & full gpt-4o
        temperature=0.9,
        max_tokens = 4096,
        # stop='hi',
    )

    dspy.settings.configure(lm=lm)
    dspy.configure(experimental=True)
    agent = Agent()

    text_desc = problem.desc.split("Input Format")[0]
    input_output_format = problem.desc.split("Input Format")[1]
    guidelines = ""
    plan = agent.get_plan(text_desc, guidelines=guidelines)
    logger.info(f"Generated Plan: {plan}")

    for _ in range(5):
        plan = agent.revise_plan(plan, text_desc, "YOU ARE WRONG!! TRY AGAIN VERBOSE")
        logger.info(f"New Generated Plan: {plan}")


    for _ in range(3):

        # first generation
        code = agent.get_code(plan, input_output_format)
        logger.info(f"Generated C++ program: {code}")
        # print(code)
        score, failed_testcases = problem.test_code(code)
        guidelines = failed_testcases
        logger.info(f"Test results - Score: {score}, Failed testcases: {failed_testcases}")
        

        # fix code loop
        for _ in range(2):
            code = agent.revise_code(plan, code, failed_testcases, input_output_format)
            # print(code)
            logger.info(f"Generated C++ program: {code}")
            score, failed_testcases = problem.test_code(code)
            guidelines = failed_testcases
            logger.info(f"Test results - Score: {score}, Failed testcases: {failed_testcases}")
            if score == 1.0:
                exit(0)

        plan = agent.revise_plan(plan, text_desc, guidelines)
        # print(plan)
        logger.info(f"New Generated Plan: {plan}")

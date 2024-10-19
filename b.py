import dspy
import logging
import random
from dspy import InputField, OutputField, Signature
from problem import Problem
from vor import Desc2PlanGenerator, UpdatePlan, Reason2CodeGenerator, Pseudo2GuidelineGenerator, SummarizeGuideline, Plan2TimeComplexityGuidelineGenerator, Plan2AlternativeSolutionsGenerator, Plan2PseudoCodeGenerator, Plan2MistakesGenerator, Plan2InvariantsGenerator, ExpandDesc

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

"""You are an expert problem solver. Your task is creating the code to solve the problem at hand in cpp.
    You are given a problem description and a sample input/output pair.
    Write code as if you are giving a coding interview and any edge cases missed will get you a rejection.
    
    
    Note:
    * Write code like you are 5
    * The variable names need to be atleast 7 character long.
    * Make sure that your proposed solution is both time and memory efficient.
    * use #include<bits/stdc++.h>
    * use #define ll long long
    * use ll instead of int
    * Surround the code with <code> tags only.
        For example:
    <code>
    ...
    </code>
    """


class GenerateCodeSignature(Signature):
    """You are an expert problem solver. Your task is creating the code to solve the problem at hand in python.
    You are given a problem description and a sample input/output pair.
    Write code as if you are giving a coding interview and any edge cases missed will get you a rejection.
    
    
    Note:
    * Write code like you are 5
    * Use very descriptive variable names.
    * Make sure that your proposed solution is both time and memory efficient.
    * use #include<bits/stdc++.h>
    * use #define ll long long
    * use ll instead of int
    * Surround the code with <code> tags only.
        For example:
    <code>
    ...
    </code>
    """

    problem_description: str = InputField(format=str)
    sample_input: str = InputField(format=str)
    sample_output: str = InputField(format=str)
    
    solution: str = OutputField(
        format=str, desc="A plan for how we should go about solving this problem."
    )
    explanation: str = OutputField(
        format=str, desc="An explanation of the solution and how it corresponds to the sample input/output."
    )
    time_complexity: str = OutputField(
        format=str, desc="Give the time complexity of the solution in O(.) notation and see if it is efficient."
    )
    cpp_program: str = OutputField(format=str)


class SimpleGenerateCode(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_code = dspy.Predict(GenerateCodeSignature)

    def forward(self, problem_description, sample_input, sample_output):
        cpp_code = extract_code(
            self.generate_code(
                problem_description=problem_description,
                sample_input=sample_input,
                sample_output=sample_output,
            ).cpp_program
        )

        return dspy.Prediction(solution=cpp_code)

if __name__ == "__main__":
    problem_name = "Line of Delivery (Part 1)"
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
        temperature=0.3,
        max_tokens = 4096,
        # stop='hi',
    )

    dspy.settings.configure(lm=lm)
    dspy.configure(experimental=True)

    desc2pseudo = Desc2PlanGenerator()
    logger.info("Evaluating Simple Program on test...")
    guidelines = ""
    expand_desc = ExpandDesc()
    expanded_desc = str(expand_desc(desc=problem.desc).expanded_desc)
    logger.info(f"Expanded description: {expanded_desc}")
    response = desc2pseudo(
        problem_description=expanded_desc, 
        guidelines=guidelines
    )
    logger.info(f"Initial response: {response}")
    time_complexity_analyzer = Plan2TimeComplexityGuidelineGenerator(desc=problem.desc)
    alternative_solutions_generator = Plan2AlternativeSolutionsGenerator(desc=problem.desc)
    mistakes_generator = Plan2MistakesGenerator(desc=problem.desc.split("Constraints")[0])
    update_plan = UpdatePlan(desc=problem.desc)
    plan2invariants_generator = Plan2InvariantsGenerator(desc=problem.desc.split("Constraints")[0])
    plan = response.plan
    p_guidelines = guidelines
    summarized_guidelines = SummarizeGuideline()
    for i in range(1):
        # for stmt in problem.desc.split("Constraints")[0].split("."):
        time_complexity_response = time_complexity_analyzer(
            plan=plan,
        )
        guidelines = p_guidelines + str(time_complexity_response.time_complexity_guideline) + "\n"
        response = desc2pseudo(
            problem_description=expanded_desc, 
            guidelines=guidelines
        )
        logger.info(f"Iteration {i+1} response plan: {response.plan}")
        plan = response.plan
        alternative_solutions_response = alternative_solutions_generator(
            plan=plan,
            previous_guidelines=guidelines
        )
        guidelines = p_guidelines + str(alternative_solutions_response.alternative_solutions) + "\n"
        response = desc2pseudo(
            problem_description=expanded_desc, 
            guidelines=guidelines
        )
        logger.info(f"Iteration {i+1} response plan: {response.plan}")
        plan = response.plan
        mistakes_response = mistakes_generator(
            plan=plan,
            previous_guidelines=guidelines
        )
        guidelines = p_guidelines + str(mistakes_response.mistakes) + "\n"
        logger.info(f"Iteration {i+1} guidelines: {guidelines}")
        response = desc2pseudo(
            problem_description=expanded_desc, 
            guidelines=guidelines
        )
        logger.info(f"Iteration {i+1} response plan: {response.plan}")
        plan = response.plan
        invariants_response = plan2invariants_generator(
            plan=plan,
        )
        guidelines = p_guidelines + str(invariants_response.invariants_and_monovariants_guideline) + "\n"
        logger.info(f"Iteration {i+1} invariants and monovariants guidelines: {guidelines}")
        response = desc2pseudo(
            problem_description=expanded_desc, 
            guidelines=guidelines
        )
        logger.info(f"Iteration {i+1} response plan: {response.plan}")
        plan = response.plan
        # print(stmt)
        # up_response = update_plan(
        #     statement=stmt,
        #     plan=plan,
        # )
        # plan = up_response.improved_plan
        logger.info(f"Iteration {i+1} updated plan: {plan}")
        p_guidelines = str(summarized_guidelines(guidelines).summarized_guidelines)

        logger.info(f"Final plan: {plan}")
        plan2pseudo = Plan2PseudoCodeGenerator()
        response = plan2pseudo(
            plan=plan,
            problem_description=problem.desc,
        )
        logger.info(f"Pseudo code: {response.pseudo_code}")
        plan = response.pseudo_code
    reason2code = Reason2CodeGenerator()
    response = reason2code(
        pseudo_code=response.pseudo_code,    
        input_ouput_format=problem.desc.split("Constraints")[1],
    )
    logger.info(f"Generated C++ program: {response.cpp_program}")
    score, failed_testcases = problem.test_code(response.cpp_program)
    logger.info(f"Test results - Score: {score}, Failed testcases: {failed_testcases}")


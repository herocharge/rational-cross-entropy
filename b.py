import dspy
from dspy import InputField, OutputField, Signature
from problem import Problem

def extract_code(response: str) -> str:
    # extract all text between <code> and </code> tags
    # there might be multiple code blocks, return the first one
    print(response)
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
    with open("Hacker cup/Line by Line/statement.txt", "r") as f:
        desc = f.read()
    problem = Problem(
        desc=desc, 
        sample_in_file="Hacker cup/Line by Line/sample_in.txt", 
        sample_out_file="Hacker cup/Line by Line/sample_out.txt"
    )

    lm = dspy.Together(
        # model="meta-llama/Llama-3-70b-chat-hf", # Note: didn't find much a difference btwn mini & full gpt-4o
        model="google/gemma-2-27b-it", # Note: didn't find much a difference btwn mini & full gpt-4o
        temperature=0.1,
        max_tokens = 4096,
        # stop='hi',
    )

    dspy.settings.configure(lm=lm)
    dspy.configure(experimental=True)

    simple_program = SimpleGenerateCode()
    print(f"Evaluating Simple Program on test...")

    response = simple_program(
        problem_description=problem.desc, 
        sample_input=problem.sample_in, 
        sample_output=problem.sample_out
    )
    print(response['solution'])

    score, failed_testcases = problem.test_code(response['solution'])
    print(score, failed_testcases)



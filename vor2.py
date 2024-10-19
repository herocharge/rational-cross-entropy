import dspy
from dspy import InputField, OutputField, Signature
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
    if out == "":
        code_blocks = re.findall(r'```(.*?)```', response, re.DOTALL)
        out = code_blocks[-1] if code_blocks else ""
    return out

class Desc2PlanSignature(Signature):
    """You are a reasoning expert. You make arguments for why a particular solution is correct VERY CAREFULLY and RIGOROUSLY. 
        Your task is take a problem description and convert it into valid plan.
        
        It should be HIGH LEVEL do not bother about the syntax of the code or implementation details.
        Note:
        * The plan should be in natural language so that further reasoning can be done on it.
        * We also provide you additional guidelines that you need to keep in mind while writing the plan.
        * Based on the guidelines, decide if the solution should fundamentally change.
        * The problems are often straight forward and are not meant to be tricky, always prioritize O(1) solutions, then O(N), then O(NlogN), then O(N^2) if necessary.
        * Do not just randomly apply well known techniques, think step by step.
    """
    problem_description: str = InputField(format=str)
    guidelines: str = InputField(format=str)
    solution: str = OutputField(
        format=str, desc="A plan for how we should go about solving this problem."
    )
    explanation: str = OutputField(
        format=str, desc="An explanation of the solution and how it corresponds to the sample input/output."
    )
    time_complexity: str = OutputField(
        format=str, desc="Give the time complexity of the solution in O(.) notation and see if it is efficient."
    )
    plan: str = OutputField(format=str)

class Desc2PlanGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_plan = dspy.Predict(Desc2PlanSignature)

    def forward(self, problem_description, guidelines):
        plan = (
            self.generate_plan(
                problem_description=problem_description,
                guidelines=guidelines
            ).plan
        )

        return dspy.Prediction(plan=plan)

class Plan2CodeSignature(Signature):
    """You are an expert coder. Your task is take plan and input output format and convert it into valid c++ code.
    
    Note:
    * USE VERBOSE VARIABLE NAMES.
    * Use time efficient functions and data structures.
    * use #include<bits/stdc++.h>
    * use #define ll long long
    * use ll instead of int
    * Surround the code with <code> tags only.
        For example:
    <code>
    ...
    </code>
    """

    plan: str = InputField(format=str)
    time_complexity: str = OutputField(
        format=str, desc="Give the time complexity of the solution in O(.) notation and see if it is efficient."
    )
    cpp_program: str = OutputField(format=str)


class Plan2CodeGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_code = dspy.Predict(Plan2CodeSignature)

    def forward(self, plan, input_ouput_format: str):
        cpp_code = extract_code(
            self.generate_code(
                plan=plan,
                input_ouput_format=input_ouput_format
            ).cpp_program
        )

        return dspy.Prediction(cpp_program=cpp_code)
    
class ReviseCodeSignature(Signature):
    """You are an expert debugger. Your task is take c++ code and a plan and error messages which could include compilation errors or failed testcases or timeouts and produce fixed code.
    
    Note:
    * USE VERBOSE VARIABLE NAMES.
    * Use time efficient functions and data structures.
    * use #include<bits/stdc++.h>
    * use #define ll long long
    * use ll instead of int
    * Surround the code with <code> tags only.
        For example:
    <code>
    ...
    </code>
    """

    plan: str = InputField(format=str)
    broken_code: str = InputField(format=str)
    error: str = InputField(format=str)
    input_output_format = InputField(format=str)

    fixed_code: str = OutputField(format=str)
    
class ReviseCode(dspy.Module):
    def __init__(self):
        super().__init__()
        self.fix_code =  dspy.Predict(ReviseCodeSignature)

    def forward(self, plan, broken_code, error, input_output_format):
        fixed_code = extract_code(self.fix_code(
                plan=plan, 
                broken_code=broken_code,
                error=error,
                input_output_format=input_output_format
            ).fixed_code)
        return fixed_code
    
class RevisePlanSignature(Signature):
    """You are very busy. Your task is take a problem description, existing plan, errors caused by the plan and give a WORKING PLAN.
    
    It should be HIGH LEVEL do not bother about the syntax of the code or implementation details.
        Note:
        * The plan should be in natural language so that further reasoning can be done on it.
    """

    plan: str = InputField(format=str)
    error: str = InputField(format=str)
    problem_description: str = InputField(format=str)
    fixed_plan: str = OutputField(format=str)
    
class RevisePlan(dspy.Module):
    def __init__(self):
        super().__init__()
        self.fix_plan =  dspy.Predict(RevisePlanSignature)

    def forward(self, plan, problem_description, error):
        fixed_plan = self.fix_plan(
                plan=plan, 
                problem_description=problem_description,
                error=error,
            ).fixed_plan
        return fixed_plan
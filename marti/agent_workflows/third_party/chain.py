import ray
import random
import numpy as np
from copy import deepcopy
from typing import List
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn
from marti.verifiers.qwen.qwen_math_parser import extract_answer

from marti.worlds.third_party.base import BaseMultiAgentGame

id2template = {
    "0": {
        "generator": "Problem: {query}\n\nYou are the Solver. Please analyze the problem step by step, then provide your final answer in the form:\n\\boxed{{Your Final Answer Here}}",
        "verifier": "Problem: {query}\nSolution: {solution}\n\nYou are the Verifier. You are given a problem and its proposed solution. Please analyze the solution in detail and offer constructive feedback. Focus on correctness, clarity, and any potential improvements. Remember: Do NOT provide a direct solution yourself.",
        "refiner": "Problem: {query}\nSolution: {solution}\nCritique: {feedback}\n\nYou are the Refiner. You have a problem, a proposed solution, and critique on that solution. Your task is to improve the solution based on the critique. Please reason through any necessary revisions step by step, then present your final improved answer in the form:\n\\boxed{{Your Final Answer Here}}"
    },
    "1": {
        "generator": "You are a solution generator. Given a problem, provide a detailed solution.\nProblem: {query}\n\nProvide the following:\n1. A list of reasoning steps\n2. The proposed solution\n3. A confidence score between 0 and 1\n4. Put your final answer within \\boxed{{}}",
        "verifier": "You are a solution verifier. Analyze the proposed solution critically.\nProblem: {query}\nProposed Solution: {solution}\n\nVerify the solution and provide:\n1. Whether it's correct (true/false)\n2. List any issues found\n3. Suggested improvements\n4. A verification confidence score between 0 and 1",
        "refiner": "You are a solution refiner. Improve the solution based on verification feedback.\nProblem: {query}\nOriginal Solution: {solution}\nVerification Feedback: {feedback}\n\nProvide:\n1. A refined solution\n2. List of improvements made\n3. Final confidence score between 0 and 1\n4. Put your final answer within \\boxed{{}}"
    },
    "2": {
        "generator": "{query}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "verifier": "You are the verifier. Your task is to analyze the following solution to a problem and give detailed, constructive feedback.\n\nProblem: {query}\n\nProposed Solution: {solution}\n\nYour goals:\n1. Carefully assess whether the solution is logically correct.\n2. Do NOT solve the problem yourself or suggest a correct solution.\n3. Provide a step-by-step reasoning of your assessment.\n4. Conclude with your final judgment clearly indicated using \\boxed{{1}} (correct) or \\boxed{{0}} (incorrect).\n\nMake sure your feedback explains *why* the solution is or isn't correct.",
        "refiner": "You are tasked with revising a draft solution to a problem based on the provided critique.\nYour goal is to produce an improved, clearer, and more accurate solution that directly addresses the feedback.\n\nProblem: {query}\n\nOriginal Solution: {solution}\n\nCritique: {feedback}\n\nYour task:\n1. Carefully read and understand the critique.\n2. Revise the original solution to correct any mistakes, address feedback, and enhance clarity.\n3. If the critique is incorrect or misleading, you may explain why and offer a reasoned alternative.\n4. Reason step by step as you refine the solution.\n\nPresent the final revised solution, and put your final boxed answer (e.g. the numeric or final result) within \\boxed{{}}.",
    },
    "3": {
        "generator": """You are a rigorous mathematician. Carefully solve the following mathematics problem step-by-step. Clearly state your reasoning and calculation details.

### Problem:
{query}

### Requirements:
- Provide a clear, structured solution.
- Explicitly state all assumptions and intermediate reasoning steps.
- Reason step by step, and put your final answer within \\boxed{{}}.

### Let's think step by step""",
        "verifier": """You are an expert mathematical reviewer tasked with critically examining another mathematician's solution. Your job is NOT to solve the problem again from scratch but to deeply analyze the provided solution.

### Original Problem:
{query}

### Proposed Solution by Previous Mathematician:
{solution}

### Tasks for Reflection:
1. Identify clearly if any mistakes, logical flaws, or missing details exist in the proposed solution.
2. Discuss explicitly why each identified part is incorrect or incomplete.
3. Suggest how the reasoning or calculation should be improved to arrive at a correct solution.

### Reflection and critique (step by step)""",
        "refiner": """You are a senior mathematician synthesizing insights from multiple sources to produce a highly accurate, complete, and refined solution.

### Original Problem:
{query}

### Initial Proposed Solution:
{solution}

### Expert Reflection & Critique:
{feedback}

### Your Tasks:
- Carefully review the initial solution along with the reflection provided.
- Correct any mistakes and explicitly address all concerns raised in the critique.
- Provide a comprehensive, corrected, and refined solution, clearly highlighting improvements and fixes.
- Reason step by step, and put your final answer within \\boxed{{}}.

### Final Refined Solution (step-by-step reasoning)"""
    },
    "4": {
        "generator": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. Your role is NOT predefined. Instead, you should carefully read the discussion context, then decide and clearly state your next action. Possible actions include but are not limited to:
- Propose a new detailed solution or reasoning steps.
- Critically analyze, find errors or limitations in previous answers.
- Reflect or summarize previous arguments, then propose improvements or corrections.
- Combine multiple previous solutions and critiques into a refined solution.

### Problem:
{query}
Please reason step by step, and put your final answer within \\boxed{{}}.


### Discussion History:
(There is no history yet)

### Your Response:
First explicitly state your chosen action in brackets, e.g. [Critique], [Solution], [Reflection], or [Combination].  
Then, clearly provide your detailed response step-by-step. Please put your final answer within \\boxed{{}}.""",
        "verifier": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. Your role is NOT predefined. Instead, you should carefully read the discussion context, then decide and clearly state your next action. Possible actions include but are not limited to:
- Propose a new detailed solution or reasoning steps.
- Critically analyze, find errors or limitations in previous answers.
- Reflect or summarize previous arguments, then propose improvements or corrections.
- Combine multiple previous solutions and critiques into a refined solution.

### Problem:
{query}
Please reason step by step, and put your final answer within \\boxed{{}}.


### Discussion History:
Agent 1: {solution}

### Your Response:
First explicitly state your chosen action in brackets, e.g. [Critique], [Solution], [Reflection], or [Combination].  
Then, clearly provide your detailed response step-by-step. Please put your final answer within \\boxed{{}}.""",
        "refiner": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. Your role is NOT predefined. Instead, you should carefully read the discussion context, then decide and clearly state your next action. Possible actions include but are not limited to:
- Propose a new detailed solution or reasoning steps.
- Critically analyze, find errors or limitations in previous answers.
- Reflect or summarize previous arguments, then propose improvements or corrections.
- Combine multiple previous solutions and critiques into a refined solution.

### Problem:
{query}
Please reason step by step, and put your final answer within \\boxed{{}}.


### Discussion History:
Agent 1: {solution}

Agent 2: {feedback}


### Your Response:
First explicitly state your chosen action in brackets, e.g. [Critique], [Solution], [Reflection], or [Combination].  
Then, clearly provide your detailed response step-by-step. Please put your final answer within \\boxed{{}}.""",
    },
    "5": {
        "generator": "{query}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "verifier": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. Your role is NOT predefined. Instead, you should carefully read the discussion context, then decide and clearly state your next action. Possible actions include but are not limited to:
- Propose a new detailed solution or reasoning steps.
- Critically analyze, find errors or limitations in previous answers.
- Reflect or summarize previous arguments, then propose improvements or corrections.
- Combine multiple previous solutions and critiques into a refined solution.

### Problem:
{query}
Please reason step by step, and put your final answer within \\boxed{{}}.


### Discussion History:
Agent 1: {solution}

### Your Response:
First explicitly state your chosen action in brackets, e.g. [Critique], [Solution], [Reflection], or [Combination].  
Then, clearly provide your detailed response step-by-step. Please put your final answer within \\boxed{{}}.""",
        "refiner": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. Your role is NOT predefined. Instead, you should carefully read the discussion context, then decide and clearly state your next action. Possible actions include but are not limited to:
- Propose a new detailed solution or reasoning steps.
- Critically analyze, find errors or limitations in previous answers.
- Reflect or summarize previous arguments, then propose improvements or corrections.
- Combine multiple previous solutions and critiques into a refined solution.

### Problem:
{query}
Please reason step by step, and put your final answer within \\boxed{{}}.


### Discussion History:
Agent 1: {solution}

Agent 2: {feedback}


### Your Response:
First explicitly state your chosen action in brackets, e.g. [Critique], [Solution], [Reflection], or [Combination].  
Then, clearly provide your detailed response step-by-step. Please put your final answer within \\boxed{{}}.""",
    },
    "6": {
        "generator": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. 
Your role is NOT fixed or predefined. Instead, you should carefully read the entire discussion so far, 
then decide (and explicitly state) what action would most effectively advance the problem-solving process.

Possible actions include (but are not limited to):
1. [Solution]: Propose or refine a detailed solution approach, step by step.
2. [Critique]: Identify errors or limitations in the latest proposed solution(s).
3. [Reflection]: Summarize or clarify the arguments so far, and suggest improvements.
4. [Combination]: Merge or reconcile multiple partial solutions or critiques into a consistent solution.
5. [Verification]: Double-check calculations and logical steps, aiming to confirm correctness or uncover mistakes.
6. [Explanation]: Explain key insights or a sub-problem in simpler terms, possibly assisting others' understanding.

Feel free to adopt any other action label you deem appropriate (e.g. [Hypothesis], [Exploration], etc.) 
if it better describes how you'll contribute to this collaborative effort.

---

### Problem:
{query}

Please reason step by step, and put your final answer within \\boxed{{}}.

---

### Discussion History:
(There is no history yet)

---

### Your Response:
1. On the first line, explicitly state your chosen action in brackets, e.g. [Solution], [Critique], [Reflection], etc.  
2. In the following lines, provide your detailed reasoning or analysis. Explain *why* you chose this particular action and *how* it builds on or differs from prior answers.  
3. Finally, write your proposed content or solution. If you give a final numerical or symbolic answer, place it within \\boxed{{}} at the end.

Remember, your role is flexible and should be chosen to best improve or advance the collective solution, 
rather than repeating or duplicating previous steps.""" ,
        "verifier": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. 
Your role is NOT fixed or predefined. Instead, you should carefully read the entire discussion so far, 
then decide (and explicitly state) what action would most effectively advance the problem-solving process.

Possible actions include (but are not limited to):
1. [Solution]: Propose or refine a detailed solution approach, step by step.
2. [Critique]: Identify errors or limitations in the latest proposed solution(s).
3. [Reflection]: Summarize or clarify the arguments so far, and suggest improvements.
4. [Combination]: Merge or reconcile multiple partial solutions or critiques into a consistent solution.
5. [Verification]: Double-check calculations and logical steps, aiming to confirm correctness or uncover mistakes.
6. [Explanation]: Explain key insights or a sub-problem in simpler terms, possibly assisting others' understanding.

Feel free to adopt any other action label you deem appropriate (e.g. [Hypothesis], [Exploration], etc.) 
if it better describes how you'll contribute to this collaborative effort.

---

### Problem:
{query}

Please reason step by step, and put your final answer within \\boxed{{}}.

---

### Discussion History:
Agent 1: {solution}

---

### Your Response:
1. On the first line, explicitly state your chosen action in brackets, e.g. [Solution], [Critique], [Reflection], etc.  
2. In the following lines, provide your detailed reasoning or analysis. Explain *why* you chose this particular action and *how* it builds on or differs from prior answers.  
3. Finally, write your proposed content or solution. If you give a final numerical or symbolic answer, place it within \\boxed{{}} at the end.

Remember, your role is flexible and should be chosen to best improve or advance the collective solution, 
rather than repeating or duplicating previous steps.""" ,
        "refiner": """### Context
You are one of several collaborative agents working together to solve complex mathematical problems. 
Your role is NOT fixed or predefined. Instead, you should carefully read the entire discussion so far, 
then decide (and explicitly state) what action would most effectively advance the problem-solving process.

Possible actions include (but are not limited to):
1. [Solution]: Propose or refine a detailed solution approach, step by step.
2. [Critique]: Identify errors or limitations in the latest proposed solution(s).
3. [Reflection]: Summarize or clarify the arguments so far, and suggest improvements.
4. [Combination]: Merge or reconcile multiple partial solutions or critiques into a consistent solution.
5. [Verification]: Double-check calculations and logical steps, aiming to confirm correctness or uncover mistakes.
6. [Explanation]: Explain key insights or a sub-problem in simpler terms, possibly assisting others' understanding.

Feel free to adopt any other action label you deem appropriate (e.g. [Hypothesis], [Exploration], etc.) 
if it better describes how you'll contribute to this collaborative effort.

---

### Problem:
{query}

Please reason step by step, and put your final answer within \\boxed{{}}.

---

### Discussion History:
Agent 1: {solution}

Agent 2: {feedback}


---

### Your Response:
1. On the first line, explicitly state your chosen action in brackets, e.g. [Solution], [Critique], [Reflection], etc.  
2. In the following lines, provide your detailed reasoning or analysis. Explain *why* you chose this particular action and *how* it builds on or differs from prior answers.  
3. Finally, write your proposed content or solution. If you give a final numerical or symbolic answer, place it within \\boxed{{}} at the end.

Remember, your role is flexible and should be chosen to best improve or advance the collective solution, 
rather than repeating or duplicating previous steps.""" ,
    },
    "7": {
        "generator": """### Context
You are one of several agents collaborating to solve mathematical problems. Your role is dynamic. Carefully review the discussion history, then select and declare your action. Choose from:
- **[Solution]**: Propose a new step-by-step solution. Explicitly state theorems/rules used.
- **[Critique]**: Identify errors/gaps in prior answers. Quote specific lines, explain issues, and provide corrected reasoning.
- **[Reflection]**: Summarize key points from the history. Highlight unresolved issues or conflicting conclusions.
- **[Combination]**: Merge valid components from previous attempts into a coherent solution. Cite contributions from each agent.

### Problem:
{query}
*Present your final answer within \boxed{{}}.*

### Discussion History:
(There is no history yet.)

### Your Response:
[Action]  // Explicitly state [Solution/Critique/Reflection/Combination]

**Step-by-Step Analysis:**
1. [Start by addressing the core objective or error from the discussion history]
2. [Apply relevant mathematical principles/theorems with justification]
3. [If critiquing: "Agent X claimed [...] but [...] because [...]"]
4. [For combinations: "Integrating Agent Y's correction [...] and Agent Z's method [...]"]
5. [Include verification step to confirm validity]

**Final Answer**
\boxed{{...}}  // Place final answer here after full derivation""",
        "verifier": """### Context
You are one of several agents collaborating to solve mathematical problems. Your role is dynamic. Carefully review the discussion history, then select and declare your action. Choose from:
- **[Solution]**: Propose a new step-by-step solution. Explicitly state theorems/rules used.
- **[Critique]**: Identify errors/gaps in prior answers. Quote specific lines, explain issues, and provide corrected reasoning.
- **[Reflection]**: Summarize key points from the history. Highlight unresolved issues or conflicting conclusions.
- **[Combination]**: Merge valid components from previous attempts into a coherent solution. Cite contributions from each agent.

### Problem:
{query}
*Present your final answer within \boxed{{}}.*

### Discussion History:
Agent 1: {solution}


### Your Response:
[Action]  // Explicitly state [Solution/Critique/Reflection/Combination]

**Step-by-Step Analysis:**
1. [Start by addressing the core objective or error from the discussion history]
2. [Apply relevant mathematical principles/theorems with justification]
3. [If critiquing: "Agent X claimed [...] but [...] because [...]"]
4. [For combinations: "Integrating Agent Y's correction [...] and Agent Z's method [...]"]
5. [Include verification step to confirm validity]

**Final Answer**
\boxed{{...}}  // Place final answer here after full derivation""",
        "refiner": """### Context
You are one of several agents collaborating to solve mathematical problems. Your role is dynamic. Carefully review the discussion history, then select and declare your action. Choose from:
- **[Solution]**: Propose a new step-by-step solution. Explicitly state theorems/rules used.
- **[Critique]**: Identify errors/gaps in prior answers. Quote specific lines, explain issues, and provide corrected reasoning.
- **[Reflection]**: Summarize key points from the history. Highlight unresolved issues or conflicting conclusions.
- **[Combination]**: Merge valid components from previous attempts into a coherent solution. Cite contributions from each agent.

### Problem:
{query}
*Present your final answer within \boxed{{}}.*

### Discussion History:
Agent 1: {solution}

Agent 2: {feedback}

### Your Response:
[Action]  // Explicitly state [Solution/Critique/Reflection/Combination]

**Step-by-Step Analysis:**
1. [Start by addressing the core objective or error from the discussion history]
2. [Apply relevant mathematical principles/theorems with justification]
3. [If critiquing: "Agent X claimed [...] but [...] because [...]"]
4. [For combinations: "Integrating Agent Y's correction [...] and Agent Z's method [...]"]
5. [Include verification step to confirm validity]

**Final Answer**
\boxed{{...}}  // Place final answer here after full derivation"""
    }
}


class MultiAgentChain(BaseMultiAgentGame):
    def __init__(
        self,
        agent_list,
        template_id,
        sampling_params,
        *args, **kwargs
    ):
        super().__init__(agent_list=agent_list, template_id=template_id, sampling_params=sampling_params)

        role2template = id2template[str(template_id)]

        self.generator_template = role2template["generator"]
        self.verifier_template = role2template["verifier"]
        self.refiner_template = role2template["refiner"]
        self.histories = []

    def distribute_prompts(self, prompts: List[str], agent_id=0, turn_id=0) -> List[str]:
        llms = self.agent_list[agent_id]["llms"]
        tokenizer = self.agent_list[agent_id]["tokenizer"]
        is_reasoning_model = self.agent_list[agent_id]["is_reasoning_model"]
        
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in prompts
        ]

        print(f"Prepare and generate with {agent_id}")
        all_output_refs = []
        batch_size = (len(chat_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            current_prompts = chat_prompts[i *
                                           batch_size: (i + 1) * batch_size]
            if is_reasoning_model:
                current_prompts = [prompt + "<think>" for prompt in current_prompts]

            if current_prompts:
                all_output_refs.append(llm.generate.remote(
                    current_prompts,
                    sampling_params=self.sampling_params,
                ))
        all_outputs = sum(ray.get(all_output_refs), [])
        all_texts = [output.outputs[0].text for output in all_outputs]
        if is_reasoning_model:
            all_texts = [
                "<think>" + text for text in all_texts
            ]

        for idx in random.sample(list(range(len(chat_prompts))), 2):
            print(f"Prompt ({agent_id}) >>>> " + repr(chat_prompts[idx]))
            print(f"Output ({agent_id}) >>>> " + repr(all_texts[idx]))

        self.organize_responses_by_problem_and_agents(
            num_problems=len(prompts),
            all_prompts=prompts,
            all_responses=all_texts,
            agent_id=agent_id,
            turn_id=turn_id
        )

        return all_texts

    def organize_responses_by_problem_and_agents(self,
                                                 num_problems,
                                                 all_prompts,
                                                 all_responses,
                                                 agent_id,
                                                 turn_id):
        assert num_problems == len(all_prompts) == len(all_responses)
        for prob_idx in range(num_problems):
            self.histories[prob_idx].append({
                "user": all_prompts[prob_idx],
                "assistant": all_responses[prob_idx],
                "agent_id": agent_id,
                "turn_id": turn_id
            })

    def process_prompt_for_thinking_model(self, solution, agent_id):
        if self.agent_list[agent_id]["is_reasoning_model"]:
            solution = solution.split("</think>")[-1].strip().strip("</answer>").strip("<answer>")
        return solution

    def run(self, problems: List[str], *args, **kwargs):
        num_problems = len(problems)
        self.histories = [
            [] for _ in range(num_problems)
        ]

        # generate solutions
        generator_prompts = [
            self.generator_template.format(query=problem) for problem in problems
        ]

        solutions = self.distribute_prompts(
            generator_prompts, agent_id=0, turn_id=0)

        # verify solutions
        verifier_prompts = [
            self.verifier_template.format(
                query=problem,
                solution=self.process_prompt_for_thinking_model(solution, 0)
            ) for problem, solution in zip(
                problems, solutions
            )
        ]
        feedbacks = self.distribute_prompts(
            verifier_prompts, agent_id=1, turn_id=1)

        # refine solutions
        refiner_prompts = [
            self.refiner_template.format(
                query=problem,
                solution=self.process_prompt_for_thinking_model(solution, 0),
                feedback=self.process_prompt_for_thinking_model(feedback, 1)
            ) for problem, solution, feedback in zip(
                problems, solutions, feedbacks
            )
        ]
        refinements = self.distribute_prompts(
            refiner_prompts, agent_id=2, turn_id=2)

    def get_history(self):
        return self.histories

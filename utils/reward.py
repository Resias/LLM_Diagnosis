import re

def reward_accuracy(completions, answers, **kwargs):
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        # Extract the answer from the completion
        try:
            # Parse the <answer>...</answer> content from the model output
            text = completion["content"] if isinstance(completion, dict) else completion
            m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            answer = m.group(1).strip() if m else ""
            
            # Binary reward: 1 for correct, 0 for incorrect
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)
        except:
            # If we can't parse an answer, give a low reward
            rewards.append(0.0)

    return rewards

def reward_format(completions, **kwargs):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c) else 0.0 for c in completions]
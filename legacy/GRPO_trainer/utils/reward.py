import re

def reward_accuracy(completions, answers, **kwargs):
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        try:
            print(f'completion[content] : {completion["content"]}')
            text = completion["content"] if isinstance(completion, dict) else completion
            m = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
            answer = m.group(1).strip() if m else ""
            is_correct = answer.strip().lower() == correct_answer.strip().lower()
            reward = 1.0 if is_correct else 0.0
            rewards.append(reward)
        except Exception as e:
            print(f"[Reward Parse Error] {e}")
            rewards.append(0.0)
    return rewards

def reward_format(completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            text = completion["content"] if isinstance(completion, dict) else completion
            pattern = r"^<think>.*?</think><answer>.*?</answer>$"
            rewards.append(1.0 if re.match(pattern, text.strip(), re.DOTALL) else 0.0)
        except Exception as e:
            print(f"[Format Parse Error] {e}")
            rewards.append(0.0)
    return rewards
import re
import math
import random
from typing import Dict, Any


class EnhancedReviewRewardFunction:
    # Regex to extract the <think> block
    THINK_REGEX = re.compile(
        r'(?is)<think>\s*(.*?)\s*</think>'
    )
    # Regex to extract the specified sections (Summary, Strengths, Weaknesses)
    SECTION_REGEX = re.compile(
        r'(?ms)^#{2,3}\s*(Summary|Strengths|Weaknesses)\s*\n(.*?)(?=\n#{2,}\s*\S+|$)'
    )
    # Regex to extract the rating
    RATING_REGEX = re.compile(r'(?i)rating[:\s]*(\d+(?:\.\d+)?)', re.IGNORECASE | re.DOTALL)

    def __init__(self,
                 rating_temperature: float = 0.5,
                 max_rating_error: float = 2.0,
                 format_weight: float = 0.5,
                 section_weight: float = 0.2,
                 depth_weight: float = 0.3,
                 rating_weight: float = 1.0):
        """
        Param:
        rating_temperature: Temperature parameter of scoring consistency 
                            (the smaller the value, the stricter the penalty).
        max_rating_error: Maximum allowable rating error (zero points if exceeded).
        format_weight, section_weight, depth_weight, rating_weight:
                         Weights for each component of the total reward.
        """
        self.rating_temperature = rating_temperature
        self.max_rating_error = max_rating_error
        self.weights = {
            'format': format_weight,
            'section': section_weight,
            'depth': depth_weight,
            'rating': rating_weight
        }

    def _extract_pred_rating(self, response: str) -> float:
        """
        Extract the predicted rating from the entire response using a 
        precompiled RATING_REGEX. Returns -1 if not found.
        """
        # match = self.RATING_REGEX.search(response)
        match = self.RATING_REGEX.findall(response)
        if match:
            try:
                # rating = float(match.group(1))
                rating = float(match[-1])
                # Clip rating to [1, 10]
                return max(1.0, min(rating, 10.0))
            except ValueError:
                pass
        return -1

    def _rating_consistency_reward(self, pred_rating: float, human_rating: float) -> float:
        """
        Compute how consistent the predicted rating is with the human rating.
        Uses a Gaussian decay function on the absolute error.
        """

        # if int(pred_rating) not in [1, 3, 5, 6, 8, 10]:
        #     return 0.0

        absolute_error = abs(pred_rating - human_rating)

        # If error exceeds max_rating_error, reward = 0
        if absolute_error > self.max_rating_error:
            return 0.0

        # Gaussian similarity: exp(-error^2 / (2 * sigma^2))
        sigma = self.rating_temperature
        similarity = math.exp(-(absolute_error ** 2) / (2 * (sigma ** 2)))

        similarity = max(0.0, min(similarity, 1.0))
        return similarity

    def _format_reward(self, review_text: str) -> float:
        """
        A simple "format" reward based on whether mandatory sections exist.
        You could make this more elaborate (e.g., checking headings, bullet points, etc.).
        """
        think_match = self.THINK_REGEX.search(review_text)
        think_exists = bool(think_match)
        # think_exists = "</think>" in review_text

        sections = {}
        for section_name, content in self.SECTION_REGEX.findall(review_text):
            sections[section_name.lower()] = content.strip()

        required_parts = {
            'think': think_exists,
            'summary': bool(sections.get('summary')),
            'strengths': bool(sections.get('strengths')),
            'weaknesses': bool(sections.get('weaknesses'))
        }
        missing_count = sum(
            1 for present in required_parts.values() if not present)
        missing_penalty = - missing_count * 0.25
        return missing_penalty

    def _depth_reward(self, response: str, sections: Dict[str, str], paper_data: Dict[str, Any]) -> float:
        """
        Reward for mentioning key elements from the paper and providing suggestions in the weakness section.
        """
        depth_score = 0.0

        # Make a one-time lowercase version for all substring checks.
        response_lower = response.lower()
        key_elements = paper_data.get('key_elements', [])
        for element in key_elements:
            if element.lower() in response_lower:
                depth_score += 0.05

        # Check if "weakness" section contains suggestions.
        weakness_section = sections.get('weakness', "").lower()
        if weakness_section:
            if re.search(r'(should\s+improve|recommend\s+to|suggest\s+that)',
                         weakness_section, flags=re.IGNORECASE):
                depth_score += 0.1

        return depth_score

    def __call__(self,
                 response: str,
                 human_rating: float) -> Dict[str, float]:
        """
        Calculate and return all rewards in a dictionary.
        {
            'total_reward': float in [0, 1],
            'rating_reward': float,
            'format_reward': float,
            'section_reward': float,
            'depth_reward': float,
            'pred_rating': float,
            'human_rating': float
        }
        """
        # Extract rating (quick single regex search).
        pred_rating = self._extract_pred_rating(response)

        # Compute sub-rewards
        rating_reward = self._rating_consistency_reward(
            pred_rating, human_rating)
        format_reward = self._format_reward(response)
        # depth_reward = self._depth_reward(response, sections, paper_data)

        # 4) Weighted sum
        total_reward = (
            self.weights['rating'] * rating_reward +
            self.weights['format'] * format_reward
            # self.weights['depth']   * depth_reward
        )

        # Clip or rescale total reward to [0, 1] if desired
        total_reward = max(-1, min(total_reward, 1.0))

        return {
            'total_reward': total_reward,
            'rating_reward': rating_reward,
            'format_reward': format_reward,
            'pred_rating': pred_rating,
            'human_rating': human_rating
        }

# Instantiate the reward function.
reward_fn = EnhancedReviewRewardFunction(
    rating_temperature=1.2, # 1.5 originally
    max_rating_error=1.0    # 1.0 originally
)

def group_review_reward_fn(generated_texts, golden_answers):
    """
    Computes rewards for a batch of generated texts using an enhanced review reward function,
    then adjusts rewards by grouping texts with the same predicted rating and awarding a bonus
    based on text length within each group.

    For each (generated_text, golden_answer) pair:
      - Prepend "<think>" to the text.
      - Compute the initial reward using EnhancedReviewRewardFunction, which returns a dictionary:
            {
                'total_reward': total_reward,
                'rating_reward': rating_reward,
                'format_reward': format_reward,
                'pred_rating': pred_rating,
                'human_rating': human_rating
            }
    
    Then, the results are grouped by pred_rating. For each group, the texts are compared by length.
    Within a group:
      - Determine the minimum (min_len) and maximum (max_len) text lengths.
      - Award a bonus to each text such that the longest text gets a bonus of max_bonus (0.2),
        and any other text receives a bonus of:
            bonus = max_bonus * (text_length - min_len) / (max_len - min_len)
      - In the case where all texts in a group have the same length (i.e. max_len == min_len),
        assign the full bonus (max_bonus) to each text.

    The bonus is added to the 'total_reward' of each corresponding result.
    
    Returns:
        A list of reward dictionaries (with updated 'total_reward') corresponding to each input text.
    """
    
    
    # First, compute initial rewards and keep track of the modified text.
    results_with_text = []
    for text, answer in zip(generated_texts, golden_answers):
        modified_text = "<think>" + text
        result = reward_fn(response=modified_text, human_rating=float(answer))
        results_with_text.append({
            'text': modified_text,
            'result': result
        })
    
    # Group the results by their predicted rating.
    groups = {}
    for item in results_with_text:
        pred = item['result']['pred_rating']
        groups.setdefault(pred, []).append(item)

    total_texts = len(generated_texts)
    group_sizes = {pred: len(group) for pred, group in groups.items()}

    # Define the maximum bonus.
    max_bonus = 0.0 # 0.2 originally
    max_diversity_bonus = 0.2
    
    # Process each group to adjust the rewards based on text length.
    for pred_rating, group_items in groups.items():
        # diversity bonus for this rating group
        # smaller groups get higher bonus
        group_ratio = group_sizes[pred_rating] / total_texts
        diversity_bonus = max_diversity_bonus * (1 - group_ratio)

        # Compute lengths of texts in the current group.
        lengths = [len(item['text']) for item in group_items]
        min_len = min(lengths)
        max_len = max(lengths)
        length_range = max_len - min_len
        # Award bonus for each text in the group.
        for item in group_items:
            current_len = len(item['text'])
            # Handle the case with no variation in length.
            if max_len == min_len:
                bonus = 0.0
            else:
                bonus = max_bonus * (current_len - min_len) / length_range
            # Update the total_reward with the bonus.
            item['result']['total_reward'] += bonus # + diversity_bonus


    
    # Reconstruct the final list of results in the original order.
    final_rewards = [item['result']["total_reward"] for item in results_with_text]
    return final_rewards
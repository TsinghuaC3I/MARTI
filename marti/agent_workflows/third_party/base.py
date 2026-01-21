from typing import List

class BaseMultiAgentGame:
    def __init__(self,
                 agent_list,
                 template_id,
                 sampling_params,
                 *args, **kwargs):
        self.agent_list = agent_list
        self.sampling_params = sampling_params
        self.template_id = template_id
        self.num_agents = len(agent_list)
    
    def run(self, problems: List[str], *args, **kwargs):
        pass
    
    def get_history(self):
        """
        [
            [{}, {}, {}], agent 0 - turn 0, turn 1, turn 2
            [{}, {}, {}], agent 1 - turn 0, turn 1, turn 2
            [{}], agent 2 - turn 0
        ]
        """
        pass
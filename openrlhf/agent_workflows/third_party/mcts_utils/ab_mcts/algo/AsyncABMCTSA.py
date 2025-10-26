import treequest as tq
import copy
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, Generic, List, Literal, Optional, Tuple, TypeVar, Union

from treequest.algos.ab_mcts_a.prob_state import NodeProbState, PriorConfig
from treequest.algos.base import Algorithm
from treequest.algos.tree import Node, Tree
from treequest.types import GenerateFnType, StateScoreType
from treequest.algos.ab_mcts_a.algo import ABMCTSAAlgoState
import time
import asyncio

StateT = TypeVar("StateT")

class AsyncABMCTSA(tq.ABMCTSA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def step(
        self,
        state: ABMCTSAAlgoState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> ABMCTSAAlgoState:
        if not inplace:
            state = copy.deepcopy(state)

        if len(state.all_rewards_store) == 0:
            for action in generate_fn:
                state.all_rewards_store[action] = []

        if not state.tree.root.children:
            await self._expand_node(state, state.tree.root, generate_fn)
            return state

        node = state.tree.root
        while node.children:
            node, action_used = await self._select_child(state, node, generate_fn)
            if action_used is not None:
                return state

        # Expansion phase
        await self._expand_node(state, node, generate_fn)
        return state
    async def _select_child(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ):
        thompson_state = state.thompson_states.get_or_create(
            node,
            list(generate_fn.keys()),
        )
        selection = thompson_state.select_next(state.all_rewards_store)

        if isinstance(selection, str):
            new_node = await self._generate_new_child(state, node, generate_fn, selection)
            return new_node, selection
        else:
            # Otherwise, we return the existing child with that index
            if selection >= len(node.children):
                raise RuntimeError(
                    f"Something went wrong in ABMCTSA algorithm: selected index {selection} is out of bounds."
                )

            return node.children[selection], None
        
    async def _expand_node(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ):
        thompson_state = state.thompson_states.get_or_create(
            node,
            list(generate_fn.keys()),
        )

        selection = thompson_state.select_next(state.all_rewards_store)

        # Ensure we get a string action name, not an index
        if not isinstance(selection, str):
            raise RuntimeError(
                f"Something went wrong in ABMCTSA algorithm: selection should always be str when the expansion is from the leaf node, whle got {selection}"
            )
        
        new_node = await self._generate_new_child(state, node, generate_fn, selection)
        return new_node, selection

    async def _generate_new_child(
        self,
        state: ABMCTSAAlgoState,
        node: Node,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        action: str,
    ):
        """
        Generate a new child node asynchronously (return task).
        """
        node_state = None if node.is_root() else node.state

        # task = asyncio.create_task(generate_fn[action](node_state))

        # async def _wrap():
        # new_state, new_score = await task
        new_state, new_score = await generate_fn[action](node_state, action)


        # Add new node to the tree
        new_node = state.tree.add_node((new_state, new_score), node)

        # Update Thompson state
        thompson_state = state.thompson_states.get(node)
        if thompson_state:
            thompson_state.register_new_child_node(
                action, new_node, self.model_selection_strategy
            )
        else:
            raise RuntimeError(
                f"Internal Error in ABMCTSA: thompson_state should not be None for node {node}"
            )

        # Backpropagate
        self._backpropagate(state, new_node, new_score, action)
        return new_node

        # return asyncio.create_task(_wrap())

        
    ## ** V1    
    # async def step(
    #     self,
    #     state: ABMCTSAAlgoState,
    #     generate_fn: Mapping[str, GenerateFnType[StateT]],
    #     inplace: bool = False,
    # ) -> ABMCTSAAlgoState:
    #     """
    #     Perform one Async_step of the Thompson Sampling MCTS algorithm.

    #     Args:
    #         state: Current algorithm state
    #         generate_fn: Mapping of action names to generation functions

    #     Returns:
    #         Updated algorithm state
    #     """
    #     if not inplace:
    #         state = copy.deepcopy(state)

    #     # initialize all_rewards_store
    #     if len(state.all_rewards_store) == 0:
    #         for action in generate_fn:
    #             state.all_rewards_store[action] = []

    #     # If the tree is empty (only root), expand the root
    #     if not state.tree.root.children:
    #         await self._expand_node(state, state.tree.root, generate_fn)
    #         return state

    #     # Run one simulation step
    #     node = state.tree.root

    #     # Selection phase: traverse tree until we reach a leaf node or need to create a new node
    #     while node.children:
    #         node, action_used = await self._select_child(state, node, generate_fn)

    #         # If action is not None, it means we've generated a new node
    #         if action_used is not None:
    #             return state

    #     # Expansion phase: expand leaf node
    #     await self._expand_node(state, node, generate_fn)

    #     return state


    # async def _select_child(
    #     self,
    #     state: ABMCTSAAlgoState,
    #     node: Node,
    #     generate_fn: Mapping[str, GenerateFnType[StateT]],
    # ) -> Tuple[Node, Optional[str]]:
    #     """
    #     Select a child node using Thompson Sampling.

    #     Args:
    #         state: Current algorithm state
    #         node: Node to select child from
    #         generate_fn: Mapping of action names to generation functions

    #     Returns:
    #         Tuple of (selected node, action if new node was generated)
    #     """
    #     # Get or create thompson state for this node
    #     thompson_state = state.thompson_states.get_or_create(
    #         node,
    #         list(generate_fn.keys()),
    #     )

    #     # Ask for next node or action using Thompson Sampling
    #     selection = thompson_state.select_next(state.all_rewards_store)

    #     # If string returned, we need to generate a new node with that action
    #     if isinstance(selection, str):
    #         new_node = await self._generate_new_child(state, node, generate_fn, selection)
    #         return new_node, selection
    #     else:
    #         # Otherwise, we return the existing child with that index
    #         if selection >= len(node.children):
    #             raise RuntimeError(
    #                 f"Something went wrong in ABMCTSA algorithm: selected index {selection} is out of bounds."
    #             )
    #         return node.children[selection], None

    # async def _expand_node(
    #     self,
    #     state: ABMCTSAAlgoState,
    #     node: Node,
    #     generate_fn: Mapping[str, GenerateFnType[StateT]],
    # ) -> Tuple[Node, str]:
    #     """
    #     Expand a leaf node by generating a new child.

    #     Args:
    #         state: Current algorithm state
    #         node: Node to expand
    #         generate_fn: Mapping of action names to generation functions

    #     Returns:
    #         Tuple of (new node, action used)
    #     """
    #     # Create thompson state for this node if it doesn't exist
    #     thompson_state = state.thompson_states.get_or_create(
    #         node,
    #         list(generate_fn.keys()),
    #     )

    #     # Get action to use for generating child
    #     selection = thompson_state.select_next(state.all_rewards_store)

    #     # Ensure we get a string action name, not an index
    #     if not isinstance(selection, str):
    #         raise RuntimeError(
    #             f"Something went wrong in ABMCTSA algorithm: selection should always be str when the expansion is from the leaf node, whle got {selection}"
    #         )

    #     new_node = await self._generate_new_child(state, node, generate_fn, selection)
    #     return new_node, selection    

    # async def _generate_new_child(
    #     self,
    #     state: ABMCTSAAlgoState,
    #     node: Node,
    #     generate_fn: Mapping[str, GenerateFnType[StateT]],
    #     action: str,
    # ) -> Node:
    #     """
    #     Generate a new child node using the specified action.

    #     Args:
    #         state: Current algorithm state
    #         node: Parent node
    #         generate_fn: Mapping of action names to generation functions
    #         action: Name of action to use for generation

    #     Returns:
    #         Newly created node
    #     """
    #     # 测试1：单个调用的时间分布
    #     start_time = time.time()
    #     cpu_start = time.process_time()
    #     # Generate new state and score using the selected action
    #     node_state = None if node.is_root() else node.state
    #     new_state, new_score = await generate_fn[action](node_state)

    #     # Add new node to the tree
    #     new_node = state.tree.add_node((new_state, new_score), node)

    #     # Update Thompson state with the new node
    #     thompson_state = state.thompson_states.get(node)
    #     if thompson_state:
    #         thompson_state.register_new_child_node(
    #             action, new_node, self.model_selection_strategy
    #         )
    #     else:
    #         raise RuntimeError(
    #             f"Internal Error in ABMCTSA: thompson_state should not be None for node {node}"
    #         )

    #     # Backpropagate the score through the parents
    #     self._backpropagate(state, new_node, new_score, action)

    #     # wall_time = time.time() - start_time
    #     # cpu_time = time.process_time() - cpu_start
    #     # # 分析结果
    #     # cpu_ratio = cpu_time / wall_time if wall_time > 0 else 0
        
    #     # print(f"Wall time: {wall_time:.3f}s")
    #     # print(f"CPU time: {cpu_time:.3f}s") 
    #     # print(f"CPU ratio: {cpu_ratio:.2f}")
        
    #     # if cpu_ratio > 0.8:
    #     #     print("📊 判断：CPU 密集型")
    #     #     print("原因：CPU 时间占总时间的 80% 以上")
    #     # elif cpu_ratio < 0.2:
    #     #     print("🌐 判断：I/O 密集型")
    #     #     print("原因：CPU 时间占总时间不到 20%，大部分时间在等待")
    #     # else:
    #     #     print("🔄 判断：混合型")
    #     #     print("原因：CPU 和 I/O 时间都比较显著")

    #     return new_node
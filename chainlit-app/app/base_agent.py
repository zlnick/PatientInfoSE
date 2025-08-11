# base_agent.py
from abc import ABC, abstractmethod
import chainlit as cl

class BaseAgent(ABC):
    """Agent基类，所有自定义Agent需继承并实现核心方法"""
    def __init__(self, name: str, description: str):
        self.name = name  # Agent名称（用于调度时识别）
        self.description = description  # Agent功能描述（用于计划生成）
    
    @abstractmethod
    async def run(self, input_data: dict, context: dict, client) -> dict:
        """
        核心执行方法
        :param input_data: 输入参数（字典格式，便于扩展）
        :param context: 上下文数据（包含历史对话、临时变量等）
        :param client: LLM客户端（复用全局客户端）
        :return: 执行结果（需包含status和output字段）
        """
        pass

    async def log_step(self, content: str):
        """统一的步骤日志输出（集成Chainlit Step）"""
        async with cl.Step(name=self.name, type="agent") as step:
            step.output = content
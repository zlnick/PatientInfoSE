import json
import iris as irisnative
import datetime

class IRISContextManager:
    def __init__(self, host, port, namespace, username, password, global_name="ChatSession"):
        self.global_name = global_name
        self.connection = irisnative.createConnection(host, port, namespace, username, password)
        self.iris = irisnative.createIRIS(self.connection)

    def _now(self):
        return datetime.datetime.utcnow().isoformat() + "Z"

    def create_session(self, session_id, meta=None):
        """新建一个对话会话（覆盖同名session）"""
        doc = {
            "session_id": session_id,
            "history": [],
            "created_at": self._now(),
            "last_updated": self._now(),
            "meta": meta or {}
        }
        self.iris.set(json.dumps(doc), self.global_name, session_id)
        return doc

    def get_session(self, session_id):
        """获取整个会话JSON文档（不存在则返回None）"""
        doc_str = self.iris.get(self.global_name, session_id)
        if doc_str:
            return json.loads(doc_str)
        return None

    def append_history(self, session_id, role, content):
        """向历史追加一条消息"""
        doc = self.get_session(session_id)
        if not doc:
            raise ValueError(f"Session {session_id} 不存在")
        doc["history"].append({
            "role": role,
            "content": content,
            "ts": self._now()
        })
        doc["last_updated"] = self._now()
        self.iris.set(json.dumps(doc), self.global_name, session_id)

    def get_history(self, session_id):
        """获取指定session的全部对话历史列表"""
        doc = self.get_session(session_id)
        return doc["history"] if doc else []

    def update_meta(self, session_id, meta: dict):
        """批量更新会话meta信息"""
        doc = self.get_session(session_id)
        if not doc:
            raise ValueError(f"Session {session_id} 不存在")
        doc["meta"].update(meta)
        doc["last_updated"] = self._now()
        self.iris.set(json.dumps(doc), self.global_name, session_id)

    def delete_session(self, session_id):
        """彻底删除整个会话"""
        self.iris.kill(self.global_name, session_id)

# ======== 用法举例 ==========
if __name__ == "__main__":
    ctx = IRISContextManager(
        host="localhost",
        port=1980,
        namespace="MCP",
        username="superuser",
        password="SYS"
    )
    session = ctx.create_session("sid001", "u001", meta={"topic": "demo"})
    ctx.append_history("sid001", "user", "你好！")
    ctx.append_history("sid001", "assistant", "您好，请问需要什么帮助？")
    print(ctx.get_history("sid001"))
    ctx.update_meta("sid001", {"foo": "bar"})
    print(ctx.get_session("sid001"))
    ctx.delete_session("sid001")
